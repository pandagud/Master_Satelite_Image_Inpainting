import torch
from torch import nn
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from src.dataLayer import makeMasks
from pathlib import Path
import torchvision.transforms as transforms
from src.shared.modelUtility import modelHelper
from src.models.UnetPartialConvModel import PartialConv2d
from src.models.loss import CalculateLoss
import os


class trainInpainting():
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
        self.save_error_step = config.save_error_step
        self.config = config
        if config.run_polyaxon:
            self.localdir = config.data_path
            self.output_path =config.output_path
        else:
            self.localdir = Path().absolute().parent
            self.output_path= Path().absolute().parent
        self.modelOutputPath = Path.joinpath(self.output_path, 'models')
        self.ImageOutputPath = Path.joinpath(self.output_path,'images')
        self.trainMode = config.trainMode
        self.modelName = config.model_name
        self.run_TCI = config.run_TCI
        self.save_model_step = config.save_model_step

    def image_tensor_batch_to_list_of_pil_images(self,image_batch, resize_resolution=None):
        """Creates a list of PIL images from a PyTorch tensor batch of 3-channel images.
        Creates a list of PIL images from a PyTorch tensor batch of 3-channel images.
        Args:
            image_batch: PyTorch tensor image batch.
            resize_resolution: Resolution which PIL images will be resized to.
        Returns:
            image_pil_list: List of PIL images.
        """
        # Ensure that there is a batch dimension
        if len(image_batch.shape) < 4:  # If there is only a single image in the batch
            image_batch = image_batch.unsqueeze(0)  # Add extra dimension (batch size dimension)

        # Loop over the batch and append pil images to list
        image_pil_list = []
        num_images = image_batch.shape[0]
        for i in range(num_images):
            image_tensor = image_batch[i, :, :, :]
            image_pil = transforms.ToPILImage()(image_tensor.cpu()).convert('RGB')
            #if resize_resolution is not None:
            #   image_pil = image_pil.resize(resize_resolution, Image.CUBIC)
            image_pil_list.append(image_pil)

        return image_pil_list

    def tensor_to_numpy(self,image_batch):
        images = []
        for i in image_batch:
            image = i.detach().cpu().numpy()
            image_numpy = image.astype(np.uint8)
            images.append(image_numpy)
        return images

    def show_tensor_images(self, image_tensorReal, image_tensorFake, image_tensorMasked,
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
        image_grid = make_grid(image_unflat1[:self.batchSize * 3], nrow=self.batchSize)
        plt.imshow(image_grid.permute(1, 2, 0).squeeze())
        plt.show()

    def trainGAN(self):
        gen = self.generator().to(self.device)
        gen_opt = torch.optim.Adam(gen.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
        disc = self.discriminator().to(self.device)
        disc_opt = torch.optim.Adam(disc.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
        filename = Path.joinpath(self.modelOutputPath, self.modelName + '_Errors_' + str(self.batchSize) + '.txt')
        criterionBCE = nn.BCELoss().cuda()
        criterionMSE = nn.L1Loss().cuda()

        # Loss function
        # Moves vgg16 model to gpu, used for feature map in loss function
        loss_func = CalculateLoss(self.config).cuda()
        print("Setup loss function...")
        cur_step = 0

        discriminator_loss = []
        generator_loss = []
        generator_loss_BCE = []

        loadAndAgumentMasks = makeMasks.MaskClass(self.config,rand_seed=None)

        # måske nn.Conv2d med vægte ikke virker når vi bruger partconv2d, i så fald måske tilføje
        # or isinstance(m,partConv2d) og læg partconv2d et sted hvor den er accessible.
        def weights_init(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m,PartialConv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m,nn.LayerNorm):
                torch.nn.init.normal_(m.weight, 0.0, 0.02)
                torch.nn.init.constant_(m.bias, 0)

        def weights_initOld(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m,PartialConv2d):
                torch.nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                torch.nn.init.normal_(m.weight, 0.0, 0.02)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m,nn.LayerNorm):
                torch.nn.init.normal_(m.weight, 0.0, 0.02)
                torch.nn.init.constant_(m.bias, 0)


        gen = gen.apply(weights_init)
        disc = disc.apply(weights_init)
        for epoch in range(self.epochs):
            # Dataloader returns the batches

            for real  in tqdm(self.dataloader,position=0,leave=True,disable=self.config.run_polyaxon):
                masks = loadAndAgumentMasks.returnTensorMasks(self.batchSize)
                masks = torch.from_numpy(masks)
                masks = masks.type(torch.cuda.FloatTensor)
                masks = 1 - masks
                masks.to(self.device)

                Sar = real[1].to(self.device)
                real = real[0].to(self.device)
                real = real.type(torch.FloatTensor).to(self.device)

                ## Update discriminator ##
                disc.zero_grad()
                fake_noise = torch.mul(real, masks)
                fake = gen(fake_noise, masks)
                disc_fake_pred = disc(fake.detach())
                disc_fake_loss = criterionBCE(disc_fake_pred, torch.zeros_like(disc_fake_pred))
                disc_real_pred = disc(real)
                disc_real_loss = criterionBCE(disc_real_pred, torch.ones_like(disc_real_pred))
                disc_loss = (disc_fake_loss + disc_real_loss)/2

                gen_score_fakes = disc_fake_pred.mean().item()
                disc_score_reals = disc_real_pred.mean().item()

                # Keep track of the average discriminator loss
                discriminator_loss.append(disc_loss.item())
                # Update gradients
                disc_loss.backward()
                # Update optimizer
                disc_opt.step()

                ## Update generator ##
                gen.zero_grad()
                fake_2 = gen(fake_noise, masks)
                disc_fake_pred2 = disc(fake_2)
                #Calculate loss
                gen_lossMSE = criterionMSE(real, fake_2)
                gen_loss_Adversarial = criterionBCE(disc_fake_pred2, torch.ones_like(disc_real_pred))

                #Add heavy penalty to pixels underneath the mask, ie, try not to make it mode_collaps?
                masks = 1-masks
                real_masked_area = torch.mul(real,masks)
                fake_masked_area = torch.mul(fake_2,masks)
                #gen_loss_Inpainted_area = criterionMSE(real_masked_area,fake_masked_area)
                gen_loss = gen_lossMSE + gen_loss_Adversarial #+ (gen_loss_Inpainted_area*5)

                gen_score_fakes1 = disc_fake_pred2.mean().item()
                # få lavet en loss function, der penalizer pixels ændret udenfor maske
                # + regner MSE/L1 på alle pixels
                gen_loss.backward()
                gen_opt.step()

                # Keep track of the average generator loss
                generator_loss.append(gen_loss.item())
                generator_loss_BCE.append(gen_loss_Adversarial.item())
                ## Visualization code ##
                #modelHelper.save_tensor_single(real[0],Path.joinpath(self.ImageOutputPath, 'epoch_' + str(epoch) + '.tiff'))
                if cur_step % self.save_model_step == 0 and cur_step > 0 and self.trainMode == False:

                    #if not training, it means we are messing around testing stuff, so no need to save model
                    #and losses
                    print(
                        f"Step {cur_step}: Generator loss: {gen_loss.item()}, discriminator loss: {disc_loss.item()}")

                    # Save loss from generator and discriminator to a file, and reset them, to avoid the list perpetually growing
                    # Name of file = model name + batch_size +
                    discriminator_loss = [sum(discriminator_loss) / len(discriminator_loss)]
                    generator_loss = [sum(generator_loss) / len(generator_loss)]
                    generator_loss_BCE = [sum(generator_loss_BCE) / len(generator_loss_BCE)]

                    self.show_tensor_images(fake_2, real, fake_noise)

                    #If in train mode, it should not display images at xx display steps, but only save the model and
                    #and losses during training
                cur_step += 1
            if self.config.run_polyaxon and epoch % 5 == 0:
                metrics = {}
                modelHelper.saveMetrics(metrics,'G_loss',generator_loss[-1],self.config.polyaxon_experiment,epoch)
                modelHelper.saveMetrics(metrics, 'G_BCE_loss', generator_loss_BCE[-1],self.config.polyaxon_experiment,epoch)
                modelHelper.saveMetrics(metrics,'D_loss',discriminator_loss[-1],self.config.polyaxon_experiment,epoch)
                modelHelper.saveMetrics(metrics, 'Disc guess on reals', disc_score_reals, self.config.polyaxon_experiment,epoch)
                modelHelper.saveMetrics(metrics, 'Disc guess on fakes', gen_score_fakes, self.config.polyaxon_experiment,
                                    epoch)
                modelHelper.saveMetrics(metrics, 'Updated disc guess on fakes', gen_score_fakes1, self.config.polyaxon_experiment,
                                    epoch)

            if epoch % self.save_model_step == 0 and self.trainMode == True:
                saveString = 'Epoch Number: ' + str(epoch) + ' Generator loss: ' + str(generator_loss[-1]) + '\n' + 'Generator loss BCE: ' + str(generator_loss_BCE[-1]) + '\n' + 'Discriminator loss: ' + str(discriminator_loss[-1]) + '\n' + 'Disc guess on reals: ' + str(disc_score_reals) + ' Disc guess on fakes: ' + str(gen_score_fakes) + ' Updated disc guess on fakes: ' + str(gen_score_fakes1) + '\n'
                modelHelper.saveToTxt(filename, saveString)
                name = str(self.modelName) + '_' + str(epoch)
                path_to_model = modelHelper.saveModel(name, self.modelOutputPath, gen, self.modelName)
                modelHelper.save_tensor_batch(real, fake_noise, fake_2, self.batchSize,
                                              Path.joinpath(self.ImageOutputPath, 'epoch_' + str(epoch)))
            elif epoch % self.save_error_step == 0 and self.trainMode == True:
                saveString = 'Epoch Number: ' + str(epoch) +'\n' + ' Generator loss: ' + str(
                        generator_loss[-1]) + '\n' + 'Generator loss BCE: ' + str(
                        generator_loss_BCE[-1]) + '\n' + 'Discriminator loss: ' + str(
                        discriminator_loss[-1]) + '\n' + 'Disc guess on reals: ' + str(disc_score_reals) + ' Disc guess on fakes: ' + str(
                        gen_score_fakes) + ' Updated disc guess on fakes: ' + str(gen_score_fakes1) + '\n'
                modelHelper.saveToTxt(filename, saveString)
        return path_to_model


# Kommer ikke til at du, da denne training phase, kører på random noise, og ikke på maskerede satelit billeder
# https://docs.fast.ai/migrating_pytorch
# device = torch.device("cuda:0" if (torch.cuda.is_available() and numberGPU > 0) else "cpu")
# learner = GANLearner.wgan(self.dataloader, self.generator, self.discriminator, opt_func=Adam, cbs=CudaCallback)
##Using CudaCallBack if we use normal dataloaders, if we use fastAI, no need for this callback
# learner.recorder.train_metrics = True #pas, bør returnere metrics for hvordan træningen gik?
# learner.recorder.valid_metrics = False
# learner.fit(self.epochs, self.lr) #wd? cbs?
# learner.show_results(max_n=9, ds_idx=0)
##Outputs
# learner.predict(self.testImageDataloader) #At whatever index the test images is
# learner.show_results()

# Training
# learner.save() Can save model and optimizer state
# learner.load() load model and optimizer state

    def trainTemporalGAN(self):
        gen = self.generator().to(self.device)
        gen_opt = torch.optim.Adam(gen.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
        disc = self.discriminator().to(self.device)
        disc_opt = torch.optim.Adam(disc.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
        #display_step = 4
        criterionBCE = nn.BCELoss().cuda()
        criterionMSE = nn.MSELoss().cuda()
        #display_step = 5
        cur_step = 0

        discriminator_loss = []
        generator_loss = []
        generator_loss_BCE = []

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
            for temp0,temp1,temp2,temp3,temp4 in tqdm(self.dataloader):
                masks0 = loadAndAgumentMasks.returnTensorMasks(self.batchSize)
                # masksInverted = 1-masks
                # masksInverted = torch.from_numpy(masksInverted)
                # masksInverted = masksInverted.type(torch.cuda.FloatTensor)
                # masksInverted.to(self.device)
                masks0 = torch.from_numpy(masks0)
                masks0 = masks0.type(torch.cuda.FloatTensor)
                masks0 = 1 - masks0

                masks1 = loadAndAgumentMasks.returnTensorMasks(self.batchSize)
                masks1 = torch.from_numpy(masks1)
                masks1 = masks1.type(torch.cuda.FloatTensor)
                masks1 = 1 - masks1

                masks2 = loadAndAgumentMasks.returnTensorMasks(self.batchSize)
                masks2 = torch.from_numpy(masks2)
                masks2 = masks2.type(torch.cuda.FloatTensor)
                masks2 = 1 - masks2

                masks3 = loadAndAgumentMasks.returnTensorMasks(self.batchSize)
                masks3 = torch.from_numpy(masks3)
                masks3 = masks3.type(torch.cuda.FloatTensor)
                masks3 = 1 - masks3

                masks4 = loadAndAgumentMasks.returnTensorMasks(self.batchSize)
                masks4 = torch.from_numpy(masks4)
                masks4 = masks4.type(torch.cuda.FloatTensor)
                masks4 = 1 - masks4

                masks =torch.cat((masks0,masks1,masks2,masks3,masks4),1).to(self.device)
                real = torch.cat((temp0[0],temp1[0],temp2[0],temp3[0],temp4[0],),1).to(self.device)
                #real = real.to(self.device)
                #t = torch.cuda.get_device_properties(0).total_memory
                #c = torch.cuda.memory_cached(0)
                #a = torch.cuda.memory_allocated(0)
                #print(t)
                #print(c)
                #print(a)
                ## Update discriminator ##
                disc_opt.zero_grad()
                # lav om så den kører på masker
                fake_noise = torch.mul(real, masks)
                fake = gen(fake_noise, masks)
                disc_fake_pred = disc(fake.detach())
                disc_fake_loss = criterionBCE(disc_fake_pred, torch.zeros_like(disc_fake_pred))
                disc_real_pred = disc(real)
                disc_real_loss = criterionBCE(disc_real_pred, torch.ones_like(disc_real_pred))
                disc_loss = (disc_fake_loss + disc_real_loss) / 2

                # Keep track of the average discriminator loss
                discriminator_loss.append(disc_loss.item())
                # Update gradients
                disc_loss.backward(retain_graph=True)
                # Update optimizer
                disc_opt.step()

                ## Update generator ##
                gen_opt.zero_grad()
                # fake_noise_2 = real*masksInverted
                fake_2 = gen(fake_noise, masks)
                disc_fake_pred = disc(fake_2)
                gen_lossMSE = criterionMSE(real, fake_2)
                gen_loss_Adversarial = criterionBCE(disc_fake_pred, torch.ones_like(disc_real_pred))
                gen_loss = gen_lossMSE + gen_loss_Adversarial
                # få lavet en loss function, der penalizer pixels ændret udenfor maske
                # + regner MSE/L1 på alle pixels
                gen_loss.backward()
                gen_opt.step()

                # Keep track of the average generator loss
                generator_loss.append(gen_loss.item())
                generator_loss_BCE.append(gen_loss_Adversarial.item())

                ## Visualization code ##
                if cur_step % self.save_model_step == 0 and cur_step > 0 and self.trainMode == False:

                    #if not training, it means we are messing around testing stuff, so no need to save model
                    #and losses
                    print(
                        f"Step {cur_step}: Generator loss: {gen_loss.item()}, discriminator loss: {disc_loss.item()}")

                    # Save loss from generator and discriminator to a file, and reset them, to avoid the list perpetually growing
                    # Name of file = model name + batch_size +
                    discriminator_loss = [sum(discriminator_loss) / len(discriminator_loss)]
                    generator_loss = [sum(generator_loss) / len(generator_loss)]
                    generator_loss_BCE = [sum(generator_loss_BCE) / len(generator_loss_BCE)]

                    self.show_tensor_images(fake_2, real, fake_noise)

                    #If in train mode, it should not display images at xx display steps, but only save the model and
                    #and losses during training
                elif cur_step % self.save_model_step == 0 and cur_step > 0 and self.trainMode == True:
                    #save model
                    torch.save(gen.state_dict(),
                               Path.joinpath(self.modelOutputPath, self.modelName + '_' + str(epoch) + '.pt'))

                    # Save loss from generator and discriminator to a file, and reset them, to avoid the list perpetually growing
                    # Name of file = model name + batch_size +
                    discriminator_loss = [sum(discriminator_loss) / len(discriminator_loss)]
                    generator_loss = [sum(generator_loss) / len(generator_loss)]
                    generator_loss_BCE = [sum(generator_loss_BCE)/len(generator_loss_BCE)]
                    self.saveToTxt(generator_loss_BCE, generator_loss_BCE, discriminator_loss)


                cur_step += 1
