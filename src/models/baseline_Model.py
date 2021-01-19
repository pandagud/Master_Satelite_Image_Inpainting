import matplotlib.pyplot as plt
from src.models.biharmonic_model import inpaint_biharmonic
from src.dataLayer import  makeMasks
from src.config_default import TrainingConfig
from pathlib import Path
from datetime import datetime
from src.shared.visualization import normalize_array
from src.shared.convert import convertToUint16
import matplotlib.image as mpimg
import cv2
import numpy as np
import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class baselineModel():
    def __init__(self, testImages,names, config):
        self.batchSize = TrainingConfig.batch_size
        self.testImages = testImages
        self.testImagesName=names
        self.config = config
        self.count= 0
        if config.run_polyaxon:
            self.imageOutputPath = config.data_path / 'data' / 'generated'
        else:
            self.imageOutputPath = Path().absolute().parent / 'data' / 'generated'
            self.imageOutputPath= Path(r"E:\Speciale\NDVIExperiment")
    def show_images(self, fakes,real,real_masked,run_TCI,name=None):
        fig, axes = plt.subplots(ncols=3, nrows=1)
        ax = axes.ravel()

        ##normalize the images so the output has the highest possible contrast
        ax[0].set_title('Original image')
        ax[0].imshow(real)

        ax[1].set_title('Masked image')
        ax[1].imshow(real_masked)

        ax[2].set_title('Inpainted result')
        ax[2].imshow(fakes)

        for a in ax:
            a.axis('off')
        fig.tight_layout()
        plt.savefig(r"E:\Speciale\NDVIExperiment\NDVI_"+str(name)+".tiff")
        self.count = self.count+1
        #plt.show()
    def storeArrayAsImage(self,path,images,index):
        images = convertToUint16(images)
        if os.path.exists(path):
            cv2.imwrite(str(Path.joinpath(path ,self.testImagesName[index] + ".tiff")),images)
        else:
            print("CreatingDir")
            os.makedirs(path)
            cv2.imwrite(str(Path.joinpath(path ,self.testImagesName[index] + ".tiff")),images)

    def storeArrayAsImageNIR(self, path, images,nirpath, nirImage, index):
        images = convertToUint16(images)
        nirImage = convertToUint16(nirImage)
        if not os.path.exists(nirpath):
            os.makedirs(nirpath)
        if os.path.exists(path):
            cv2.imwrite(str(Path.joinpath(path, self.testImagesName[index] + ".tiff")), images)
            cv2.imwrite(str(Path.joinpath(nirpath, self.testImagesName[index] + ".tiff")), nirImage)
        else:
            print("CreatingDir")
            os.makedirs(path)
            cv2.imwrite(str(Path.joinpath(path, self.testImagesName[index] + ".tiff")), images)
            cv2.imwrite(str(Path.joinpath(nirpath, self.testImagesName[index] + ".tiff")), nirImage)
    def baselineExperiment(self):

        loadAndAgumentMasks = makeMasks.MaskClass(self.config,rand_seed=None,evaluation=True)
        #If we load for each batchsize
        #masks = loadAndAgumentMasks.returnTensorMasks(self.batchSize)
        now = datetime.now()
        dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
        # Defect image over the same region in each color channel
        self.output_path = self.imageOutputPath / self.config.model_name / dt_string / 'Data'
        self.nir_output_path = self.imageOutputPath / self.config.model_name / dt_string / 'NIRData'


        index = 0
        start_time = datetime.now()
        for real in self.testImages:

            # Get masks and make tensor, set to GPU
            mask = loadAndAgumentMasks.returnMask()
            mask = mask[0, :, :]
            # Get real and set to GPU

            # Augment with masks
            # Check if this applies to  all three color channels?
            real_masked = real.copy()
            for layer in range(real_masked.shape[-1]):
                real_masked[np.where(mask)] = 0
            results = inpaint_biharmonic(real_masked, mask, multichannel=True)
            if self.config.test_mode:
                self.storeArrayAsImage(self.output_path,results,index)
            else:
                self.show_images(results, real, real_masked,self.config.run_TCI)
                results= normalize_array(results)
                real = normalize_array(real)
                real_masked = normalize_array(real_masked)
                self.show_images(results, real, real_masked, self.config.run_TCI)
            index=index+1
        end_time = datetime.now()
        time_ran = str(end_time - start_time)
        print("It took "+str(time_ran))
        print("stored at "+str(self.output_path))
        return self.output_path, time_ran

    def baselineExperimentNIR(self,nirImages):

        loadAndAgumentMasks = makeMasks.MaskClass(self.config, rand_seed=None, evaluation=True)
        # If we load for each batchsize
        # masks = loadAndAgumentMasks.returnTensorMasks(self.batchSize)
        now = datetime.now()
        dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
        # Defect image over the same region in each color channel
        self.output_path = self.imageOutputPath / 'Croatia' / dt_string / 'Data'
        self.nir_output_path = self.imageOutputPath / 'Croatia'  / dt_string / 'NIRData'

        index = 0
        start_time = datetime.now()
        for real in self.testImages:

            # Get masks and make tensor, set to GPU
            mask = loadAndAgumentMasks.returnMask(82)
            mask = mask[0, :, :]
            # Get real and set to GPU

            # Augment with masks
            # Check if this applies to  all three color channels?
            real_masked = real.copy()
            NIR_real_masked = nirImages[index]
            for layer in range(real_masked.shape[-1]):
                real_masked[np.where(mask)] = 0
            for layer in range(NIR_real_masked.shape[-1]):
                NIR_real_masked[np.where(mask)] = 0
            NIR_results = inpaint_biharmonic(NIR_real_masked, mask)
            results = inpaint_biharmonic(real_masked, mask, multichannel=True)
            if self.config.test_mode:
                self.storeArrayAsImageNIR(self.output_path, results,self.nir_output_path,NIR_results, index)
                results = normalize_array(results)
                real = normalize_array(real)
                real_masked = normalize_array(real_masked)
                self.show_images(results, real, real_masked, self.config.run_TCI,name=self.testImagesName[index])
            else:
                self.show_images(results, real, real_masked, self.config.run_TCI)
                results = normalize_array(results)
                real = normalize_array(real)
                real_masked = normalize_array(real_masked)
                self.show_images(results, real, real_masked, self.config.run_TCI)
            index = index + 1
        end_time = datetime.now()
        time_ran = str(end_time - start_time)
        print("It took " + str(time_ran))
        print("stored at " + str(self.output_path))
        return self.output_path, time_ran



