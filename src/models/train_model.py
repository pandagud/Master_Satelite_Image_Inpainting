
#from fastai.callback.all import *
#from fastai import *
#from fastai.vision import *
from fastai.vision.gan import GANLearner,Adam
from fastai.callback.data import CudaCallback

from src.models.UnetPartialConvModel import generator,discriminator
from src.config_default import TrainingConfig
import torch

class trainInpainting():
    def __init__(self, dataloader, generator, discriminator):
        self.dataloader = dataloader
        self.generator = generator
        self.discriminator = discriminator
        self.batchSize = TrainingConfig.batch_size
        self.epochs = TrainingConfig.epochs
        self.numberGPU = TrainingConfig.numberGPU #if using pure pytorch
        self.lr = TrainingConfig.lr

    def trainGAN(self):
#https://docs.fast.ai/migrating_pytorch
        #device = torch.device("cuda:0" if (torch.cuda.is_available() and numberGPU > 0) else "cpu")
        learner = GANLearner.wgan(self.dataloader, self.generator, self.discriminator, opt_func=Adam, cbs=CudaCallback)
        #Using CudaCallBack if we use normal dataloaders, if we use fastAI, no need for this callback
        learner.recorder.train_metrics = True #pas, bør returnere metrics for hvordan træningen gik?
        learner.recorder.valid_metrics = False
        learner.fit(self.epochs, self.lr) #wd? cbs?
        learner.show_results(max_n=9, ds_idx=0)
        #Training
        #learner.save() Can save model and optimizer state
        #learner.load() load model and optimizer state