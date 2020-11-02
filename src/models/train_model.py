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
        self.localdir = Path().absolute().parent
        self.modelOutputPath = Path.joinpath(self.localdir, 'models')
        self.trainMode = config.trainMode
        self.modelName = config.model_name

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
    def saveToTxt(self,generatorlossBCE, generatorLoss, discLossBCE):
        #Function to save to txt file.
        filename = Path.joinpath(self.modelOutputPath, self.modelName + '_' + str(self.batchSize) + '.txt')
        # Creates file if it does not exist, else does nothing
        filename.touch(exist_ok=True)
        # then open, write and close file again
        file = open(filename, 'a+')
        file.write('Generator loss: ' + str(generatorLoss[0]) + '\n' + 'Generator loss BCE: ' + str(
            generatorlossBCE[0]) + '\n' + 'Discriminator loss: ' + str(discLossBCE[0]) + '\n')
        file.close()

    def trainGAN(self):
        gen = self.generator().to(self.device)
        gen_opt = torch.optim.Adam(gen.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
        disc = self.discriminator().to(self.device)
        disc_opt = torch.optim.Adam(disc.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
        criterionBCE = nn.BCELoss().cuda()
        criterionMSE = nn.MSELoss().cuda()
        display_step = 5
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

                cur_batch_size = len(real)
                real = real.to(self.device)
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
                if cur_step % display_step == 0 and cur_step > 0 and self.trainMode == False:

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
                elif cur_step % display_step == 0 and cur_step > 0 and self.trainMode == True:
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

