import os
from copy import deepcopy
from random import  seed
#from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
import torch
from torchvision.utils import save_image
from pathlib import Path
#Inspiration drawn from https://github.com/MathiasGruber/PConv-Keras/blob/master/libs/util.py
import gc


class MaskClass():
    def __init__(self, height, width, channels=3, rand_seed=None):

        self.localdir = Path().absolute().parent
        self.mask_path = Path.joinpath(self.localdir, 'data\\masks\\irregular_mask\\disocclusion_img_mask')
        self.height = height
        self.width = width
        self.channels = channels
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
  #  def flow_from_directory(self, directory):
            # Get augmentend image samples
 #           generator = super().flow_from_directory(directory,target_size(256))
  #          ori = next(generator)

            # Get masks for each image sample
            #mask = np.stack([
   #             self.FindAndAugmentMask()
   #             for _ in range(ori.shape[0])], axis=0
            #)

            # Apply masks to all image sample
   #         masked = deepcopy(ori)
   #         masked[mask == 0] = 1

            # Yield ([ori, masl],  ori) training batches
            # print(masked.shape, ori.shape)
    #        gc.collect()
    #        yield [masked, mask], ori
    def AugmentMasksOnImages(self,testDataLoader):
        index = 0
        for image in testDataLoader.indices:
            mask = self.FindAndAugmentMask()
            masked = deepcopy(testDataLoader.dataset[testDataLoader.indices[index]])
            #med kopi af elemented der skal lægges maske på
            test = masked[0]
            test2 = torch.from_numpy(mask)
            #så begge er tensor, og så erstattes 0 i maskens placeringer, med 1 i billedet
            test[test2==0] = 1
            #masked = newImage
            index = index + 1
        return
#Deprecated
    def AugmentTestData(self, testDataloader):

        index = 0
        for image in testDataloader.indices:
            mask = self.FindAndAugmentMask()
            masked = deepcopy(testDataloader.dataset[testDataloader.indices[index]])
            #med kopi af elemented der skal lægges maske på
            test = masked[0]
            test2 = torch.from_numpy(mask)
            #så begge er tensor, og så erstattes 0 i maskens placeringer, med 1 i billedet
            test[test2==0] = 1
            #masked = newImage
            index = index + 1
            save_image(test,'%s/MaskedImage_%03d.tiff' %(self.filepath, index),normalize=True)
            #save_image((test, '%s/fake_samples_epoch_%03d.png' %(self.filepath, index), normalize=True)

            #masked = masked*newMaskTensor

            #masked = torch.masked_fill(masked, newMaskTensor)
            #masked[mask==0] = 1
            #måske det her kan lykkedes, hvor den overider ind i træningssæt
            #testDataloader.dataset[testDataloader.indices[index]] = masked
            #Ellers det her, hvor den gemmer det nye sæt i MaskedTrainData
    def returnTensorMasks(self,batchSize):
        mask = np.stack([
            self.FindAndAugmentMask(dilation=False)
            for _ in range(batchSize)], axis=0
        )
        return mask
        #for i in len(batchSize):
        #    mask = self.FindAndAugmentMask()
        #    test2 = torch.from_numpy(mask)

    def FindAndAugmentMask(self, rotation=True, dilation=True, cropping=True):
        ##PathToMasks = r"C:\Users\Morten From\Documents\SateliteImages\irregular_mask\mask\testing_mask_dataset"  # læg ind i filepath måske senere
        mask = cv2.imread(os.path.join(self.mask_path, np.random.choice(self.mask_files, 1, replace=False)[0]))

        # Rotation
        if rotation:
            rand = np.random.randint(-180, 180)
            M = cv2.getRotationMatrix2D((mask.shape[1] / 2, mask.shape[0] / 2), rand, 1.5)
            mask = cv2.warpAffine(mask, M, (mask.shape[1], mask.shape[0]))

        # Random dilation
        if dilation:
            rand = np.random.randint(5, 47)
            kernel = np.ones((rand, rand), np.uint8)
            mask = cv2.erode(mask, kernel, iterations=1)

           # Random cropping
        if cropping:
            x = np.random.randint(0, mask.shape[1] - self.width)
            y = np.random.randint(0, mask.shape[0] - self.height)
            mask = mask[y:y + self.height, x:x + self.width]

            #burde få den til at være channels * heigth* width
        mask = np.swapaxes(mask, 0, 2)
        mask = np.swapaxes(mask, 1, 2)
        return (mask > 1).astype(np.uint8)




    #masked = deepcopy(ori) hvor ori er et billede i den sti med test billeder
    #masked[mask == 0] = 1
    # def flow_from_directory(self, directory, mask_generator, *args, **kwargs):
    #     generator = super().flow_from_directory(directory, class_mode=None, *args, **kwargs)
    #     seed = None if 'seed' not in kwargs else kwargs['seed']
    #     while True:
    #         # Get augmentend image samples
    #         ori = next(generator)
    #
    #         # Get masks for each image sample
    #         mask = np.stack([
    #             mask_generator.sample(seed)
    #             for _ in range(ori.shape[0])], axis=0
    #         )
    #
    #         # Apply masks to all image sample
    #         masked = deepcopy(ori)
    #         masked[mask == 0] = 1
    #
    #         # Yield ([ori, masl],  ori) training batches
    #         # print(masked.shape, ori.shape)
    #         gc.collect()
    #         yield [masked, mask], ori