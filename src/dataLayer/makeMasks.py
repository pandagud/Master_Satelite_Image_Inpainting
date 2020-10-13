import os
from copy import deepcopy
from random import  seed
#from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
import torch
from torchvision.utils import save_image
from pathlib import Path
from src.config_default import TrainingConfig
#Inspiration drawn from https://github.com/MathiasGruber/PConv-Keras/blob/master/libs/util.py
import gc


class MaskClass():
    def __init__(self, rand_seed=None):

        self.localdir = Path().absolute().parent
        self.mask_path = Path.joinpath(self.localdir, 'data\\masks\\mask\\testing_mask_dataset')
        self.height = TrainingConfig.image_size
        self.width = TrainingConfig.image_size
        self.rand_seed = rand_seed
        self.maskpath = self.mask_path
        self.filepath = self.mask_path


        #Find de maske png filer der er downloadet fra NVIDIA's sæt
        #gem dem i mask_files
        self.mask_files = []
        if self.maskpath:
            filenames = [f for f in os.listdir(self.maskpath)]
            self.mask_files = [f for f in filenames if
                               any(filetype in f.lower() for filetype in ['.png'])]
            print(">> Found {} masks in {}".format(len(self.mask_files), self.filepath))

        #Hvis vi vil lave reproduceable masks
        if rand_seed:
            seed(rand_seed)

    def returnTensorMasks(self,batchSize):
        mask = np.stack([
            self.FindAndAugmentMask(rotation=False)
            for _ in range(batchSize)], axis=0
        )
        return mask

    def returnMask(self):
        mask = self.FindAndAugmentMask(rotation=False)
        return mask

    def FindAndAugmentMask(self, rotation=True):
        ##PathToMasks = r"C:\Users\Morten From\Documents\SateliteImages\irregular_mask\mask\testing_mask_dataset"  # læg ind i filepath måske senere
        mask = cv2.imread(os.path.join(self.mask_path, np.random.choice(self.mask_files, 1, replace=False)[0]))
        mask = cv2.resize(mask,(self.height, self.width))

        if rotation:
            rand = np.random.randint(-180, 180)
            M = cv2.getRotationMatrix2D((mask.shape[1] / 2, mask.shape[0] / 2), rand, 1.5)
            mask = cv2.warpAffine(mask, M, (mask.shape[1], mask.shape[0]))

        #burde få den til at være channels * heigth* width, istedet for height width channels
        mask = np.swapaxes(mask, 0, 2)
        mask = np.swapaxes(mask, 1, 2)
        mask = 1-mask
        return (mask > 1).astype(np.uint8)


    #hvis vi vil gemme billederne med masker på, kan denne metode bruges
    def AugmentTestData(self, testDataloader):

        index = 0
        for image in testDataloader.indices:
            mask = self.FindAndAugmentMask()
            masked = deepcopy(testDataloader.dataset[testDataloader.indices[index]])
            # med kopi af elemented der skal lægges maske på
            test = masked[0]
            test2 = torch.from_numpy(mask)
            # så begge er tensor, og så erstattes 0 i maskens placeringer, med 1 i billedet
            test[test2 == 0] = 1
            # masked = newImage
            index = index + 1
            save_image(test, '%s/MaskedImage_%03d.tiff' % (self.filepath, index), normalize=True)
            # save_image((test, '%s/fake_samples_epoch_%03d.png' %(self.filepath, index), normalize=True)