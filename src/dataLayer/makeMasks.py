import os
from copy import deepcopy
from random import  seed
import numpy as np
import torch
from PIL import Image
from torchvision.utils import save_image
from pathlib import Path
from src.config_default import TrainingConfig
import albumentations as A
import cv2
#Inspiration drawn from https://github.com/MathiasGruber/PConv-Keras/blob/master/libs/util.py


class MaskClass():
    def __init__(self, config,rand_seed=None,evaluation=None,noFlip=None,experiment=None):

        if config.run_polyaxon:
            self.localdir = config.data_path
        else:
            self.localdir = Path().absolute().parent
        self.mask_path = Path.joinpath(self.localdir, 'data','mask','testing_mask_dataset')
        if evaluation:
            self.mask_path = Path.joinpath(self.localdir, 'data', 'mask', 'eval_mask_dataset')
        if experiment:
            self.mask_path = Path.joinpath(self.localdir, 'data', 'mask', 'experiment_mask_dataset')
        self.height = TrainingConfig.image_size
        self.width = TrainingConfig.image_size
        self.rand_seed = rand_seed
        self.config =config
        self.maskpath = self.mask_path
        self.filepath = self.mask_path
        if noFlip:
            self.transform = A.Compose([
            ])
        else:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5)
            ])


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

    def returnTensorMasks(self,batchSize,maskIndex=None):
        if maskIndex!=None:
         mask = np.stack([
                self.FindAndAugmentMask(maskIndex)
                for _ in range(batchSize)], axis=0
            )
        else:
            mask = np.stack([
                self.FindAndAugmentMask()
                for _ in range(batchSize)], axis=0
            )
        return mask

    def returnMask(self,maskIndex=None):
        if maskIndex:
            mask = self.FindAndAugmentMask(maskIndex)
        else:
            mask = self.FindAndAugmentMask()
        return mask

    def FindAndAugmentMask(self,maskIndex=None):
        ##PathToMasks = r"C:\Users\Morten From\Documents\SateliteImages\irregular_mask\mask\testing_mask_dataset"  # læg ind i filepath måske senere
        if self.config.nir_data:
            if maskIndex!=None:
                dim_1_mask = Image.open(os.path.join(self.mask_path, self.mask_files[maskIndex]))
            else:
                dim_1_mask = Image.open(os.path.join(self.mask_path, np.random.choice(self.mask_files, 1, replace=False)[0]))
            dim_1_mask = dim_1_mask.resize((self.height, self.width))
            dim_1_mask = np.array(dim_1_mask)
            mask = np.stack((np.copy(dim_1_mask),np.copy(dim_1_mask),np.copy(dim_1_mask),np.copy(dim_1_mask)),axis=2)
        else:
            if maskIndex!=None:
                mask = Image.open(os.path.join(self.mask_path, self.mask_files[maskIndex])).convert('RGB')
            else:
                mask = Image.open(os.path.join(self.mask_path, np.random.choice(self.mask_files, 1, replace=False)[0])).convert('RGB')
            mask = mask.resize((self.height, self.width))
            mask = np.array(mask)
        mask = self.transform(image=np.array(mask))['image']
        #burde få den til at være channels * heigth* width, istedet for height width channels
        mask = np.swapaxes(mask, 0, 2)
        mask = np.swapaxes(mask, 1, 2)
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