
#from fastai.callback.all import *
#from fastai import *
#from fastai.vision import *
from fastai.vision.gan import GANLearner,Adam
from fastai.callback.data import CudaCallback

from src.models.UnetPartialConvModel import generator,discriminator
from src.config_default import TrainingConfig
import torch
from torch import nn
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from src.dataLayer import makeMasks

class trainInpainting():
    def __init__(self, dataloader, testImageDataloader, generator, discriminator):
        self.dataloader = dataloader
        self.testdataloader = testImageDataloader
        self.generator = generator
        self.discriminator = discriminator
        self.batchSize = TrainingConfig.batch_size
        self.epochs = TrainingConfig.epochs
        self.numberGPU = TrainingConfig.numberGPU #if using pure pytorch
        self.lr = TrainingConfig.lr
        self.beta1 = TrainingConfig.beta1
        self.beta2 = TrainingConfig.beta2
        self.device = TrainingConfig.device

    def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28)):
        '''
        Function for visualizing images: Given a tensor of images, number of images, and
        size per image, plots and prints the images in an uniform grid.
        '''
        image_tensor = (image_tensor + 1) / 2
        image_unflat = image_tensor.detach().cpu()
        image_grid = make_grid(image_unflat[:num_images], nrow=5)
        plt.imshow(image_grid.permute(1, 2, 0).squeeze())
        plt.show()

    def trainGAN(self):
        gen = self.generator().to(self.device)
        gen_opt = torch.optim.Adam(gen.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
        disc = self.discriminator().to(self.device)
        disc_opt = torch.optim.Adam(disc.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
        criterion = nn.BCEWithLogitsLoss()
        display_step = 500
        cur_step = 0
        #få lagt loadAndAugmentmasks på billeder i en fil for sig
        #randseed returnerer nok de samme masker
        loadAndAgumentMasks = makeMasks.MaskClass(height=256,width=256,channels=3,rand_seed=None)
        #maskpath er lige nu hardcodet i makeMasks
        #måske nn.Conv2d med vægte ikke virker når vi bruger partconv2d, i så fald måske tilføje
        #or isinstance(m,partConv2d) og læg partconv2d et sted hvor den er accessible.
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
                cur_batch_size = len(real)
                real = real.to(self.device)
                masks = loadAndAgumentMasks.returnTensorMasks()
                ## Update discriminator ##
                disc_opt.zero_grad()
                #lav om så den kører på masker

                fake_noise = real*masks
                fake = gen(fake_noise,masks)
                disc_fake_pred = disc(fake.detach())
                disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
                disc_real_pred = disc(real)
                disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))
                disc_loss = (disc_fake_loss + disc_real_loss) / 2

                # Keep track of the average discriminator loss
                mean_discriminator_loss += disc_loss.item() / display_step
                # Update gradients
                disc_loss.backward(retain_graph=True)
                # Update optimizer
                disc_opt.step()

                ## Update generator ##
                gen_opt.zero_grad()
                fake_noise_2 = real*masks
                fake_2 = gen(fake_noise_2,masks)
                disc_fake_pred = disc(fake_2)
                gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))
                gen_loss.backward()
                gen_opt.step()

                # Keep track of the average generator loss
                mean_generator_loss += gen_loss.item() / display_step

                ## Visualization code ##
                if cur_step % display_step == 0 and cur_step > 0:
                    print(
                        f"Step {cur_step}: Generator loss: {mean_generator_loss}, discriminator loss: {mean_discriminator_loss}")
                    self.show_tensor_images(fake)
                    self.show_tensor_images(real)
                    mean_generator_loss = 0
                    mean_discriminator_loss = 0
                cur_step += 1


#Kommer ikke til at du, da denne training phase, kører på random noise, og ikke på maskerede satelit billeder
#https://docs.fast.ai/migrating_pytorch
        #device = torch.device("cuda:0" if (torch.cuda.is_available() and numberGPU > 0) else "cpu")
        #learner = GANLearner.wgan(self.dataloader, self.generator, self.discriminator, opt_func=Adam, cbs=CudaCallback)
        ##Using CudaCallBack if we use normal dataloaders, if we use fastAI, no need for this callback
        #learner.recorder.train_metrics = True #pas, bør returnere metrics for hvordan træningen gik?
        #learner.recorder.valid_metrics = False
        #learner.fit(self.epochs, self.lr) #wd? cbs?
        #learner.show_results(max_n=9, ds_idx=0)
        ##Outputs
        #learner.predict(self.testImageDataloader) #At whatever index the test images is
        #learner.show_results()

        #Training
        #learner.save() Can save model and optimizer state
        #learner.load() load model and optimizer state