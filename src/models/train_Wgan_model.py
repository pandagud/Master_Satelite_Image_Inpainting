# from fastai.callback.all import *
# from fastai import *
# from fastai.vision import *
from fastai.vision.gan import GANLearner, Adam
from fastai.callback.data import CudaCallback

from src.models.UnetPartialConvModel import generator, discriminator
import torch
from torch import nn
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from src.dataLayer import makeMasks
from pathlib import Path
from torch.autograd import Variable
import torch.autograd as autograd


class trainInpaintingWgan():
    def __init__(self, dataloader, testImageDataloader, generator, discriminator, config):
        self.dataloader = dataloader
        self.testdataloader = testImageDataloader
        self.generator = generator
        self.discriminator = discriminator
        self.batchSize = config.batch_size
        self.epochs = config.epochs
        self.numberGPU = config.numberGPU  # if using pure pytorch
        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.device = config.device
        self.localdir = Path().absolute().parent
        self.modelOutputPath = Path.joinpath(self.localdir, 'models')
        self.trainMode = config.trainMode
        self.modelName = config.model_name
        self.n_critic = config.n_critic
        self.lambda_gp = config.lambda_gp
        self.save_model_step = config.save_model_step

    def show_tensor_images(self, image_tensorReal, image_tensorFake, image_tensorMasked, num_images = 12,
                           size=(3, 256, 256)):
        '''
        Function for visualizing images: Given a tensor of images, number of images, and
        size per image, plots and prints the images in an uniform grid.
        '''
        image_tensor1 = (image_tensorReal + 1) / 2
        image_unflat1 = image_tensor1.detach().cpu()
        image_tensor2 = (image_tensorFake + 1) / 2
        image_unflat2 = image_tensor2.detach().cpu()
        image_tensor3 = (image_tensorMasked + 1) / 2
        image_unflat3 = image_tensor3.detach().cpu()
        image_unflat1 = torch.cat((image_unflat1, image_unflat2, image_unflat3), dim=0)
        image_grid = make_grid(image_unflat1[:num_images * 3], nrow=num_images)
        plt.imshow(image_grid.permute(1, 2, 0).squeeze())
        plt.show()

    def saveToTxt(self,generatorloss, discLoss):
        #Function to save to txt file.
        filename = Path.joinpath(self.modelOutputPath, self.modelName + '_' + str(self.batchSize) + '.txt')
        # Creates file if it does not exist, else does nothing
        filename.touch(exist_ok=True)
        # then open, write and close file again
        file = open(filename, 'a+')
        file.write('Generator loss: ' + str(generatorloss) + '\n' + 'Discriminator loss: ' + str(discLoss) + '\n')
        file.close()

    def calc_gradient_penalty(self, netD, real_data, fake_data):
        # print real_data.size()
        alpha = torch.Tensor(np.random.random((real_data.size(0), 1, 1, 1)))
        alpha = alpha.to(self.device)

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        if self.device == 'cuda':
            interpolates = interpolates.cuda()
        interpolates = autograd.Variable(interpolates, requires_grad=True)

        disc_interpolates = netD(interpolates)

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size()).cuda(
                                      ) if self.device == 'cuda' else torch.ones(
                                      disc_interpolates.size()),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.lambda_gp
        return gradient_penalty

    def trainGAN(self):
        gen = self.generator().to(self.device)
        gen_opt = torch.optim.Adam(gen.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
        disc = self.discriminator().to(self.device)
        disc_opt = torch.optim.Adam(disc.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))

        #display_step = 5
        cur_step = 0

        loadAndAgumentMasks = makeMasks.MaskClass(rand_seed=None)

        # måske nn.Conv2d med vægte ikke virker når vi bruger partconv2d, i så fald måske tilføje
        # or isinstance(m,partConv2d) og læg partconv2d et sted hvor den er accessible.
        def weights_init(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.normal_(m.weight, 0.0, 0.02)
            if isinstance(m, nn.BatchNorm2d):
                torch.nn.init.normal_(m.weight, 0.0, 0.02)
                torch.nn.init.constant_(m.bias, 0)

        gen = gen.apply(weights_init)
        disc = disc.apply(weights_init)





        for epoch in range(self.epochs):
            # Dataloader returns the batches
            for real, _ in tqdm(self.dataloader):
                masks = loadAndAgumentMasks.returnTensorMasks(self.batchSize)
                # masksInverted = 1-masks
                # masksInverted = torch.from_numpy(masksInverted)
                # masksInverted = masksInverted.type(torch.cuda.FloatTensor)
                # masksInverted.to(self.device)

                masks = torch.from_numpy(masks)
                masks = masks.type(torch.cuda.FloatTensor)
                masks = 1 - masks
                masks.to(self.device)

                real = real.to(self.device)

                # ---------------------
                #  Train Discriminator
                # ---------------------

                disc_opt.zero_grad()

                # Generate a batch of images with mask
                Masked_fake_img = torch.mul(real, masks)
                fake_imgs = gen(Masked_fake_img, masks)

                # Real images
                real_validity = disc(real)
                # Fake images
                fake_validity = disc(fake_imgs.detach()) #Detach or not?
                # Gradient penalty
                gradient_penalty = self.calc_gradient_penalty(disc, real.data, fake_imgs.data)
                # Adversarial loss
                d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + self.lambda_gp * gradient_penalty

                d_loss.backward()
                disc_opt.step()

                gen_opt.zero_grad()

                # Train the generator every n_critic steps
                if cur_step % self.n_critic == 0:

                    # -----------------
                    #  Train Generator
                    # -----------------

                    # Generate a batch of images
                    fake_noise = torch.mul(real, masks)
                    fake_imgs = gen(fake_noise, masks)
                    # Loss measures generator's ability to fool the discriminator
                    # Train on fake images
                    fake_validity = disc(fake_imgs)
                    g_loss = -torch.mean(fake_validity)

                    g_loss.backward()
                    gen_opt.step()

                    ## Visualization code ##
                    if cur_step % self.save_model_step == 0 and cur_step > 0 and self.trainMode == False:

                        # if not training, it means we are messing around testing stuff, so no need to save model
                        # and losses
                        print(
                            f"Step {cur_step}: Generator loss: {g_loss.item()}, discriminator loss: {d_loss.item()}")

                        self.show_tensor_images(fake_imgs, real, fake_noise)

                        # If in train mode, it should not display images at xx display steps, but only save the model and
                        # and losses during training
                    elif cur_step % self.save_model_step == 0 and cur_step > 0 and self.trainMode == True:
                        # save model
                        torch.save(gen.state_dict(),
                                   Path.joinpath(self.modelOutputPath, self.modelName + '_' + str(epoch) + '.pt'))

                        # Save loss from generator and discriminator to a file, and reset them, to avoid the list perpetually growing
                        #Name of file = model name + batch_size +
                        #discriminator_loss = [sum(discriminator_loss) / len(discriminator_loss)]
                        #generator_loss = [sum(generator_loss) / len(generator_loss)]
                        #generator_loss_BCE = [sum(generator_loss_BCE) / len(generator_loss_BCE)]
                        self.saveToTxt(g_loss.item(), d_loss.item())

                cur_step += 1

