import matplotlib.pyplot as plt

from skimage import data
from skimage.restoration import inpaint
import numpy as np
from src.dataLayer import  makeMasks
from src.config_default import TrainingConfig
import torch
import tqdm

class baselineModel():
    def __init__(self, testImages, config):
        self.batchSize = TrainingConfig.batch_size
        self.testImages = testImages
        self.config = config

    def show_images(self, fakes,real,real_masked,run_TCI):
        fig, axes = plt.subplots(ncols=3, nrows=1)
        ax = axes.ravel()

        ##normalize the images so the output has the highest possible contrast
        if run_TCI==False:
            fakes = fakes * 255.0 / fakes.max()
            fakes = fakes.astype(np.uint8)

            real = real * 255.0 / real.max()
            real = real.astype(np.uint8)

            real_masked = real_masked * 255.0 / real_masked.max()
            real_masked = real_masked.astype(np.uint8)

        ax[0].set_title('Original image')
        ax[0].imshow(real)

        ax[1].set_title('Masked image')
        ax[1].imshow(real_masked)

        ax[2].set_title('Inpainted result')
        ax[2].imshow(fakes)

        for a in ax:
            a.axis('off')

        fig.tight_layout()
        plt.show()

    def baselineExperiment(self):

        loadAndAgumentMasks = makeMasks.MaskClass(rand_seed=None)
        #If we load for each batchsize
        #masks = loadAndAgumentMasks.returnTensorMasks(self.batchSize)

        # Defect image over the same region in each color channel


        for real in self.testImages:

            #Get masks and make tensor, set to GPU
            mask = loadAndAgumentMasks.returnMask()
            mask = mask[0, :, :]
            mask = 1-mask
            #Get real and set to GPU

            #Augment with masks
            #Check if this applies to  all three color channels?
            real_masked = real.copy()
            for layer in range(real_masked.shape[-1]):
                real_masked[mask==1] = 1 #or 0?
            results = inpaint.inpaint_biharmonic(real_masked, mask, multichannel=True)

            self.show_images(results, real, real_masked,self.config.run_TCI)


