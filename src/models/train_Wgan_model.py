import torch
from torch import nn
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from src.dataLayer import makeMasks
from pathlib import Path
import torch.autograd as autograd
from src.shared.modelUtility import modelHelper
from src.models.UnetPartialConvModel import PartialConv2d
from src.models.loss import CalculateLoss
#https://github.com/caogang/wgan-gp
class trainInpaintingWgan():
    def __init__(self, dataloader, testImageDataloader, generator, critic, config):
        self.dataloader = dataloader
        self.testdataloader = testImageDataloader
        self.generator = generator
        self.critic = critic
        self.batchSize = config.batch_size
        self.epochs = config.epochs
        self.epochsFrozen = config.frozenEpochs
        self.numberGPU = config.numberGPU  # if using pure pytorch
        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.device = config.device
        if config.run_polyaxon:
            self.localdir = config.data_path
            self.output_path = config.output_path
        else:
            self.localdir = Path().absolute().parent
            self.output_path = Path().absolute().parent
        self.modelOutputPath = Path.joinpath(self.output_path, 'models')
        self.ImageOutputPath = Path.joinpath(self.output_path, 'images')
        self.trainMode = config.trainMode
        self.modelName = config.model_name
        self.n_critic = config.n_critic
        self.lambda_gp = config.lambda_gp
        self.save_model_step = config.save_model_step
        self.trainWithFreeze = config.trainFrozen
        self.config=config


    def calc_gradient_penalty(self, netD, real_data, fake_data):
        #https://github.com/caogang/wgan-gp
        #https://github.com/pytorch/pytorch/issues/2534
        #alpha = torch.Tensor(np.random.random((real_data.size(0), 1, 1, 1)))
        #alpha = alpha.expand_as(real_data)
        alpha = torch.rand(self.batchSize, 1)
        test = real_data.nelement()/self.batchSize
        test = int(test)
        if self.config.nir_data:
            alpha = alpha.expand(self.batchSize, test).contiguous().view(self.batchSize, 4, 256, 256)
        else:
            alpha = alpha.expand(self.batchSize, test).contiguous().view(self.batchSize, 3, 256, 256)
        alpha = alpha.to(self.device)

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        interpolates.to(self.device)
        #tror autrgrad.variable er outdated, men er ikke sikker
        interpolates = autograd.Variable(interpolates, requires_grad=True)

        disc_interpolates = netD(interpolates)

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size()).to(self.device),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)
        new_gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.lambda_gp
        #new_gradient_penalty = torch.mean((1. - torch.sqrt(1e-8+torch.sum(gradients.view(gradients.size(0), -1)**2, dim=1)))**2)
        return new_gradient_penalty


    def trainGAN(self):
        gen = self.generator().to(self.device)
        gen_opt = torch.optim.Adam(gen.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
        critic = self.critic().to(self.device)
        critic_opt = torch.optim.Adam(critic.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))

        cur_step = 0
        loadAndAgumentMasks = makeMasks.MaskClass(self.config,rand_seed=None)

        # måske nn.Conv2d med vægte ikke virker når vi bruger partconv2d, i så fald måske tilføje
        # or isinstance(m,partConv2d) og læg partconv2d et sted hvor den er accessible.
        def weights_init(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, PartialConv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                torch.nn.init.normal_(m.weight, 0.0, 0.02)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.02)
        gen = gen.apply(weights_init)
        critic = critic.apply(weights_init)

        print("Setup loss function...")
        loss_func = CalculateLoss(config=self.config).to(self.device)


        for epoch in range(self.epochs):
            for real in tqdm(self.dataloader,position=0,leave=True,disable=self.config.run_polyaxon):

                masks = loadAndAgumentMasks.returnTensorMasks(self.batchSize)
                masks = torch.from_numpy(masks)
                masks = masks.type(torch.cuda.FloatTensor)
                masks = 1 - masks
                masks.to(self.device)


                real = real.to(self.device)

                # ---------------------
                #  Train critic
                # ---------------------
                critic.zero_grad()
                # Real images
                real_validity = critic(real)
                d_real = real_validity.mean()
                # Generate a batch of images with mask
                Masked_fake_img = torch.mul(real, masks)
                fake_imgs = gen(Masked_fake_img, masks)
                # Fake images
                fake_validity = critic(fake_imgs)  # Detach or not?
                d_fake = fake_validity.mean()

                gradient_penalty = self.calc_gradient_penalty(critic, real.data, fake_imgs.data)
                d_loss = d_fake-d_real+gradient_penalty
                d_loss.backward()

                critic_opt.step()

                # Values for txt / logging
                critic_cost = d_fake-d_real+gradient_penalty
                wasserstein_d = d_real-d_fake
                critic_score = real_validity.mean().item()
                gen_score = fake_validity.mean().item()


                # Train the generator every n_critic steps
                if cur_step % self.n_critic == 0:

                    # -----------------
                    #  Train Generator
                    # -----------------
                    gen.zero_grad()
                    # Generate a batch of images
                    fake_noise = torch.mul(real, masks)
                    fake_imgs = gen(fake_noise, masks)
                    # Loss measures generator's ability to fool the critic
                    # Train on fake images
                    fake_validity1 = critic(fake_imgs)

                    loss_dict = loss_func(fake_noise, masks, fake_imgs, real)
                    loss = 0.0

                    # sums up each loss value
                    for key, value in loss_dict.items():
                        loss += value

                    loss.backward(retain_graph=True)

                    g_loss = fake_validity1.mean()
                    #g_lossMSE = criterionMSE(real, fake_imgs)
                    #g_lossMSE.backward(retain_graph=True)

                    g_loss = -g_loss
                    g_loss.backward() #mone

                    gen_opt.step()
                    gen_cost = g_loss
                cur_step += 1
            if self.config.run_polyaxon and epoch % 5 == 0:
                metrics = {}
                for key,value in loss_dict.items():
                    modelHelper.saveMetrics(metrics, key, value.item(), self.config.polyaxon_experiment, epoch)
                modelHelper.saveMetrics(metrics,'critic cost', critic_cost,self.config.polyaxon_experiment,epoch)
                modelHelper.saveMetrics(metrics, 'Wasserstein distance', wasserstein_d, self.config.polyaxon_experiment, epoch)
                modelHelper.saveMetrics(metrics, 'Gen cost', gen_cost, self.config.polyaxon_experiment,epoch)

            if epoch % self.save_model_step == 0 and self.trainMode == True:
                name = str(self.modelName) + '_' + str(epoch)
                model_path = modelHelper.saveModel(name, self.modelOutputPath, gen, self.modelName)
                if self.config.nir_data:
                    modelHelper.save_tensor_batch_NIR(real, Masked_fake_img, fake_imgs, self.batchSize,
                                              Path.joinpath(self.ImageOutputPath, 'epoch_' + str(epoch)))
                else:
                    modelHelper.save_tensor_batch(real, Masked_fake_img, fake_imgs, self.batchSize,
                                              Path.joinpath(self.ImageOutputPath, 'epoch_' + str(epoch)))
                # Save loss from generator and critic to a file

                filename = Path.joinpath(self.modelOutputPath, self.modelName + '_' + str(self.batchSize) + 'Errors.txt')
                saveString = 'wasserStein Number: ' + str(wasserstein_d) +' Generator loss: ' + str(g_loss.item()) + '\n' + 'critic loss: ' + str(d_loss.item()) + '\n' + 'critic guess on reals: ' + str(critic_score) + ' critic guess on fakes: ' + str(gen_score) + ' Updated critic guess on fake: ' + str(gen_cost) + '\n'
                modelHelper.saveToTxt(filename, saveString)

        if self.trainWithFreeze:
            #trainFrozenModel = trainFrozenGan(self.dataloader,gen,critic,gen_opt,critic_opt, self.config)
            #trainFrozenGan.trainGAN()
            #Frys BN i encoder parts of the network
            #Bruge affine? eller sætte weight og bias til module.eval
            for name, module in gen.named_modules():
                if isinstance(module, nn.BatchNorm2d) and 'down' in name:
                    module.eval()

            for epoch in range(self.epochsFrozen):
                for real in tqdm(self.dataloader, position=0, leave=True, disable=self.config.run_polyaxon):

                    masks = loadAndAgumentMasks.returnTensorMasks(self.batchSize)
                    masks = torch.from_numpy(masks)
                    masks = masks.type(torch.cuda.FloatTensor)
                    masks = 1 - masks
                    masks.to(self.device)

                    real = real.to(self.device)

                    # ---------------------
                    #  Train critic
                    # ---------------------
                    critic.zero_grad()
                    # Real images
                    real_validity = critic(real)
                    d_real = real_validity.mean()
                    # Generate a batch of images with mask
                    Masked_fake_img = torch.mul(real, masks)
                    fake_imgs = gen(Masked_fake_img, masks)
                    # Fake images
                    fake_validity = critic(fake_imgs)  # Detach or not?
                    d_fake = fake_validity.mean()

                    gradient_penalty = self.calc_gradient_penalty(critic, real.data, fake_imgs.data)
                    d_loss = d_fake - d_real + gradient_penalty
                    d_loss.backward()

                    critic_opt.step()

                    # Values for txt / logging
                    critic_cost = d_fake - d_real + gradient_penalty
                    wasserstein_d = d_real - d_fake
                    critic_score = real_validity.mean().item()
                    gen_score = fake_validity.mean().item()

                    # Train the generator every n_critic steps
                    if cur_step % self.n_critic == 0:

                        # -----------------
                        #  Train Generator
                        # -----------------
                        gen.zero_grad()
                        # Generate a batch of images
                        fake_noise = torch.mul(real, masks)
                        fake_imgs = gen(fake_noise, masks)
                        # Loss measures generator's ability to fool the critic
                        # Train on fake images
                        fake_validity1 = critic(fake_imgs)

                        loss_dict = loss_func(fake_noise, masks, fake_imgs, real)
                        loss = 0.0

                        # sums up each loss value
                        for key, value in loss_dict.items():
                            loss += value

                        loss.backward(retain_graph=True)

                        g_loss = fake_validity1.mean()
                        # g_lossMSE = criterionMSE(real, fake_imgs)
                        # g_lossMSE.backward(retain_graph=True)

                        g_loss = -g_loss
                        g_loss.backward()  # mone

                        gen_opt.step()
                        gen_cost = g_loss
                    cur_step += 1

                if self.config.run_polyaxon and epoch % 5 == 0:
                    metrics = {}
                    for key, value in loss_dict.items():
                        modelHelper.saveMetrics(metrics, key, value.item(), self.config.polyaxon_experiment, epoch+self.epochs)
                    modelHelper.saveMetrics(metrics, 'critic cost', critic_cost, self.config.polyaxon_experiment, epoch+self.epochs)
                    modelHelper.saveMetrics(metrics, 'Wasserstein distance', wasserstein_d,
                                            self.config.polyaxon_experiment, epoch+self.epochs)
                    modelHelper.saveMetrics(metrics, 'Gen cost', gen_cost, self.config.polyaxon_experiment, epoch+self.epochs)
                if epoch % self.save_model_step == 0 and self.trainMode == True:
                    name = str(self.modelName) + '_' + str(epoch+self.epochs)
                    model_path = modelHelper.saveModel(name, self.modelOutputPath, gen, self.modelName)
                    if self.config.nir_data:
                        modelHelper.save_tensor_batch_NIR(real, Masked_fake_img, fake_imgs, self.batchSize,
                                                          Path.joinpath(self.ImageOutputPath, 'epoch_' + str(epoch)))
                    else:
                        modelHelper.save_tensor_batch(real, Masked_fake_img, fake_imgs, self.batchSize,
                                                  Path.joinpath(self.ImageOutputPath, 'epoch_' + str(epoch+self.epochs)))
                    # Save loss from generator and critic to a file

                    filename = Path.joinpath(self.modelOutputPath,
                                             self.modelName + '_' + str(self.batchSize) + 'Errors.txt')
                    saveString = 'wasserStein Number: ' + str(wasserstein_d) + ' Generator loss: ' + str(
                        g_loss.item()) + '\n' + 'critic loss: ' + str(
                        d_loss.item()) + '\n' + 'critic guess on reals: ' + str(
                        critic_score) + ' critic guess on fakes: ' + str(
                        gen_score) + ' Updated critic guess on fake: ' + str(gen_cost) + '\n'
                    modelHelper.saveToTxt(filename, saveString)



        return model_path
