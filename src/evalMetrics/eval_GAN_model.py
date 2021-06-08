import torch
import logging
import click
import os
import sys
from src.evalMetrics.FID import FIDCalculator
from src.evalMetrics.PSNR import PSNR
from src.evalMetrics.Pytorch_SSIM import ssim, SSIM
from src.evalMetrics.CC import CC
from src.evalMetrics.MAE import MSE
from src.evalMetrics.SDD import SDD
from src.evalMetrics.RMSE import RMSE
from src.evalMetrics.SSIM import SSIM_SKI
from pathlib import Path
from datetime import datetime
from src.dataLayer.importRGB import importData,TCIDatasetLoader
from src.config_default import TrainingConfig
from src.config_utillity import update_config
from src.dataLayer import makeMasks
from tqdm.auto import tqdm
from polyaxon_client.tracking import get_data_paths, get_outputs_path
from src.shared.modelUtility import modelHelper
from src.shared.evalUtility import saveEvalToTxt
from src.models.UnetPartialConvModel import generator, Wgangenerator
from src.models.UnetPartialConvModelNIR import generatorNIR,Wgangenerator
from src.shared.visualization import normalize_batch_tensor

class eval_model():
    def __init__(self, config):
        self.config=config
    def run_eval(self,output_path,store_path,model_path=None,test_dataloader=None):
        curdatLayer = importData(self.config)
        if test_dataloader is None:
            train, test_dataloader = curdatLayer.getRGBDataLoader()
            del train
        if Path.exists(Path.joinpath(output_path, self.config.model_name + '_' + str(self.config.epochs) + '.pt'))and self.config.run_polyaxon == False:
            ##Hvis det er med wgan generator, altså layernorm, indsæt Wgangenerator istedet for generator()
            if self.config.new_generator:
                gen = Wgangenerator().to(self.config.device)
            else:
                gen = generator().to(self.config.device)
            gen.load_state_dict(torch.load(Path.joinpath(output_path, self.config.model_name + '_' + str(
                self.config.epochs) + '.pt')))  ## Use epochs to identify model number
        elif Path.exists(Path(str(model_path))):
            if self.config.nir_data:
                gen = generatorNIR().to(self.config.device)
            elif self.config.new_generator:
                gen = Wgangenerator().to(self.config.device)
            else:
                gen = generator().to(self.config.device)
            print("Just loaded model from path " + str(model_path))
            gen.load_state_dict(torch.load(model_path))  ## Use epochs to identify model number
        else:
            print("Unable to find path to model")
        gen.eval()

        loadAndAgumentMasks = makeMasks.MaskClass(self.config, rand_seed=None, evaluation=True)
        names = []
        # Find names of test images, in order to save the generated files with same name, for further reference
        localImg = test_dataloader.dataset.image_list
        # Slice string to only include the name of the file, ie after the last //
        localNames = []
        # if self.config.run_polyaxon:
        #     split_path = localImg[0].split('/')  ##Linux
        # else:
        #     split_path = localImg[0].split("\\")
        # local_index= split_path.index('processed')
        # local_country= split_path[local_index+1]
        for i in localImg:
            if self.config.run_polyaxon:
                selected_image = i.split('/')[-1]  ##Linux
            else:
                selected_image = i.split("\\")[-1]
            localNames.append(selected_image)
        names = names + localNames

        print("Found this many names " + str(len(names)))

        current_number = 0

        if not os.path.exists(Path.joinpath(output_path, self.config.model_name)):
            os.makedirs(Path.joinpath(output_path, self.config.model_name))

        now = datetime.now()
        dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")

        local_test_path = output_path / self.config.model_name / dt_string / 'Data'
        local_test_nir_path = output_path / self.config.model_name / dt_string / 'DataNir'
        local_store_path = store_path / self.config.model_name / dt_string / 'stored_Data'
        os.makedirs(local_test_path)
        os.makedirs(local_store_path)
        os.makedirs(local_test_nir_path)
        start_time = datetime.now()
        for real,sar in tqdm(test_dataloader,disable=self.config.run_polyaxon):
            masks = loadAndAgumentMasks.returnTensorMasks(self.config.batch_size)
            masks = torch.from_numpy(masks)
            masks = masks.type(torch.cuda.FloatTensor)
            masks = 1 - masks
            masks.to(self.config.device)

            real = real.to(self.config.device)
            fake_masked_images = torch.mul(real, masks)
            generated_images = gen(fake_masked_images, masks)
            image_names = names[current_number:current_number + self.config.batch_size]
            current_number = current_number + self.config.batch_size  ## Change naming to include all names
            # modelHelper.save_tensor_batch(generated_images,fake_masked_images,config.batch_size,path)
            for index, image in enumerate(generated_images):
                namePath = Path.joinpath(local_test_path, image_names[index])
                if self.config.nir_data:
                    modelHelper.save_tensor_single_NIR(image,Path.joinpath(local_test_path,image_names[index]),Path.joinpath(local_test_nir_path,image_names[index]),raw=True)
                else:
                    modelHelper.save_tensor_single(image, Path.joinpath(local_test_path, image_names[index]), raw=True)
        end_time = datetime.now()
        time_ran = str(end_time - start_time)
        # create dataloader with generated images
        generated_images_dataloader = curdatLayer.getGeneratedImagesDataloader(local_test_path)
        #print("generated image 429 "+str(generated_images_dataloader.dataset.image_list[429]))
        #print ("test image 429 "+str(test_dataloader.dataset.image_list[429]))

        # calculate FID
        if self.config.nir_data: # Loader test_dataloader in for NIR igen da den skal have 3 channels og ikke 4.
            train, test_dataloader = curdatLayer.getRGBDataLoader()
            del train
        FID_Value = FIDCalculator(test_dataloader, generated_images_dataloader,
                                  len(test_dataloader) * self.config.batch_size,
                                  self.config.batch_size, self.config).get_FID_scores()

        # Calculate PSNR and SSIM
        dataloader_iterator = iter(generated_images_dataloader)
        #dataloader_iterator = iter(test_dataloader)
        maeValues = []
        sddValues = []
        ssimscikitValues= []
        SSIMValues = []
        psnrValues = []
        CCValues = []
        rmseValues=[]
        # loop to calculate PSNR and SSIM for all test and generated images.
        count = 0
        for i,images_real in enumerate(test_dataloader):
            try:
                images_generated = next(dataloader_iterator)
            except StopIteration:
                dataloader_iterator = iter(generated_images_dataloader)
                images_generated = next(dataloader_iterator)

            for index2 in range(self.config.batch_size):
                psnrValues.append(PSNR().__call__(images_real[index2], images_generated[index2]))
                if psnrValues[-1]<3:
                    print(str(psnrValues[-1]))
                    modelHelper.save_tensor_single(normalize_batch_tensor(images_real[index2]),Path.joinpath(local_store_path,str(i)+'_'+str(count)+'_real.tiff'))
                    modelHelper.save_tensor_single(normalize_batch_tensor(images_generated[index2]), Path.joinpath(local_store_path,str(i)+'_'+str(count)+'gen.tiff'))
                CCValues.append(CC().__call__(images_real[index2], images_generated[index2]))
                maeValues.append(MSE().__call__(images_real[index2], images_generated[index2]))
                sddValues.append(SDD.__call__(images_real[index2], images_generated[index2]))
                ssimscikitValues.append(SSIM_SKI.__call__(images_real[index2], images_generated[index2]))
                image1 = images_real[index2].unsqueeze(0)
                image2 = images_generated[index2].unsqueeze(0)
                SSIMValues.append(ssim(image1, image2))
                rmseValues.append(RMSE.__call__(images_real[index2], images_generated[index2]))
            count= count+1
        meanMAE = sum(maeValues) / len(maeValues)
        minMAE = min(maeValues)
        maxMAE = max(maeValues)

        meanSDD = sum(sddValues) / len(sddValues)
        minSDD = min(sddValues)
        maxSDD = max(sddValues)

        meanPSNR = sum(psnrValues) / len(psnrValues)
        minPSNR = min(psnrValues)
        maxPSNR = max(psnrValues)

        meanSSIM = sum(SSIMValues) / len(SSIMValues)
        minSSIM = min(SSIMValues)
        maxSSIM = max(SSIMValues)

        meanSCISSIM = sum(ssimscikitValues) / len(ssimscikitValues)
        minSCISSIM = min(ssimscikitValues)
        maxSCISSIM = max(ssimscikitValues)

        meanCC = sum(CCValues) / len(CCValues)
        minCC = min(CCValues)
        maxCC = max(CCValues)

        meanRMSE=sum(rmseValues) / len(rmseValues)
        minRMSE = min(rmseValues)
        maxRMSE = max(rmseValues)
        # Save final results of evaluation metrics
        saveEvalToTxt(self.config.model_name, meanMAE,minMAE,maxMAE,meanSDD,minSDD,maxSDD, meanSSIM.item(),
                      minSSIM.item(),
                      maxSSIM.item(), meanSCISSIM,minSCISSIM,maxSCISSIM, meanPSNR, minPSNR, maxPSNR,meanCC, minCC, maxCC,meanRMSE,minRMSE,maxRMSE, FID_Value, time_ran, local_store_path)
        # Clean
        modelHelper.clearFolder(local_test_path.parent)

    def run_eval_TCI(self, output_path, store_path, model_path=None, test_dataloader=None):
        curdatLayer = TCIDatasetLoader(self.config)
        if test_dataloader is None:
            train, test_dataloader = curdatLayer.getTCIDataloder()
            del train
        if Path.exists(Path.joinpath(output_path, self.config.model_name + '_' + str(
                self.config.epochs) + '.pt')) and self.config.run_polyaxon == False:
            ##Hvis det er med wgan generator, altså layernorm, indsæt Wgangenerator istedet for generator()
            if self.config.new_generator:
                gen = Wgangenerator().to(self.config.device)
            else:
                gen = generator().to(self.config.device)
            gen.load_state_dict(torch.load(Path.joinpath(output_path, self.config.model_name + '_' + str(
                self.config.epochs) + '.pt')))  ## Use epochs to identify model number
        elif Path.exists(Path(str(model_path))):
            if self.config.nir_data:
                gen = generatorNIR().to(self.config.device)
            elif self.config.new_generator:
                gen = Wgangenerator().to(self.config.device)
            else:
                gen = generator().to(self.config.device)
            print("Just loaded model from path " + str(model_path))
            gen.load_state_dict(torch.load(model_path))  ## Use epochs to identify model number
        else:
            print("Unable to find path to model")
        gen.eval()

        loadAndAgumentMasks = makeMasks.MaskClass(self.config, rand_seed=None, evaluation=True)
        names = []
        # Find names of test images, in order to save the generated files with same name, for further reference
        localImg = test_dataloader.dataset.imgs
        # Slice string to only include the name of the file, ie after the last //
        localNames = []
        # if self.config.run_polyaxon:
        #     split_path = localImg[0].split('/')  ##Linux
        # else:
        #     split_path = localImg[0].split("\\")
        # local_index= split_path.index('processed')
        # local_country= split_path[local_index+1]
        for i in localImg:
            if self.config.run_polyaxon:
                selected_image = i[0].split('/')[-1]  ##Linux
            else:
                selected_image = i[0].split("\\")[-1]
            localNames.append(selected_image)
        names = names + localNames

        print("Found this many names " + str(len(names)))

        current_number = 0

        if not os.path.exists(Path.joinpath(output_path, self.config.model_name)):
            os.makedirs(Path.joinpath(output_path, self.config.model_name))

        now = datetime.now()
        dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")

        local_test_path = output_path / self.config.model_name / dt_string / 'Data'
        local_test_nir_path = output_path / self.config.model_name / dt_string / 'DataNir'
        local_store_path = store_path / self.config.model_name / dt_string / 'stored_Data'
        os.makedirs(local_test_path)
        os.makedirs(local_store_path)
        os.makedirs(local_test_nir_path)
        start_time = datetime.now()
        for real,target in tqdm(test_dataloader, disable=self.config.run_polyaxon):
            masks = loadAndAgumentMasks.returnTensorMasks(self.config.batch_size)
            masks = torch.from_numpy(masks)
            masks = masks.type(torch.cuda.FloatTensor)
            masks = 1 - masks
            masks.to(self.config.device)

            real = real.to(self.config.device)
            fake_masked_images = torch.mul(real, masks)
            generated_images = gen(fake_masked_images, masks)
            image_names = names[current_number:current_number + self.config.batch_size]
            current_number = current_number + self.config.batch_size  ## Change naming to include all names
            # modelHelper.save_tensor_batch(generated_images,fake_masked_images,config.batch_size,path)
            for index, image in enumerate(generated_images):
                namePath = Path.joinpath(local_test_path, image_names[index])
                if self.config.nir_data:
                    modelHelper.save_tensor_single_NIR(image, Path.joinpath(local_test_path, image_names[index]),
                                                       Path.joinpath(local_test_nir_path, image_names[index]), raw=True)
                else:
                    modelHelper.save_tensor_single(image, Path.joinpath(local_test_path, image_names[index]))
        end_time = datetime.now()
        time_ran = str(end_time - start_time)
        # create dataloader with generated images
        generated_images_dataloader = curdatLayer.getGeneratedImagesDataloader(local_test_path.parent)
        # print("generated image 429 "+str(generated_images_dataloader.dataset.image_list[429]))
        # print ("test image 429 "+str(test_dataloader.dataset.image_list[429]))

        # calculate FID
        if self.config.nir_data:  # Loader test_dataloader in for NIR igen da den skal have 3 channels og ikke 4.
            train, test_dataloader = curdatLayer.getRGBDataLoader()
            del train
        FID_Value = FIDCalculator(test_dataloader, generated_images_dataloader,
                                  len(test_dataloader) * self.config.batch_size,
                                  self.config.batch_size, self.config,TCI=True).get_FID_scores()

        # Calculate PSNR and SSIM
        dataloader_iterator = iter(generated_images_dataloader)
        # dataloader_iterator = iter(test_dataloader)
        maeValues = []
        sddValues = []
        ssimscikitValues = []
        SSIMValues = []
        psnrValues = []
        CCValues = []
        rmseValues = []
        # loop to calculate PSNR and SSIM for all test and generated images.
        count = 0
        for i, images_real in enumerate(test_dataloader):
            try:
                images_generated = next(dataloader_iterator)
            except StopIteration:
                dataloader_iterator = iter(generated_images_dataloader)
                images_generated = next(dataloader_iterator)

            for index2 in range(self.config.batch_size):
                psnrValues.append(PSNR().__call__(images_real[0][index2], images_generated[0][index2],max_value=255))
                CCValues.append(CC().__call__(images_real[0][index2], images_generated[0][index2]))
                maeValues.append(MSE().__call__(images_real[0][index2], images_generated[0][index2]))
                sddValues.append(SDD.__call__(images_real[0][index2], images_generated[0][index2]))
                ssimscikitValues.append(SSIM_SKI.__call__(images_real[0][index2], images_generated[0][index2]))
                image1 = images_real[0][index2].unsqueeze(0)
                image2 = images_generated[0][index2].unsqueeze(0)
                SSIMValues.append(ssim(image1, image2))
                rmseValues.append(RMSE.__call__(images_real[0][index2], images_generated[0][index2]))
            count = count + 1
        meanMAE = sum(maeValues) / len(maeValues)
        minMAE = min(maeValues)
        maxMAE = max(maeValues)

        meanSDD = sum(sddValues) / len(sddValues)
        minSDD = min(sddValues)
        maxSDD = max(sddValues)

        meanPSNR = sum(psnrValues) / len(psnrValues)
        minPSNR = min(psnrValues)
        maxPSNR = max(psnrValues)

        meanSSIM = sum(SSIMValues) / len(SSIMValues)
        minSSIM = min(SSIMValues)
        maxSSIM = max(SSIMValues)

        meanSCISSIM = sum(ssimscikitValues) / len(ssimscikitValues)
        minSCISSIM = min(ssimscikitValues)
        maxSCISSIM = max(ssimscikitValues)

        meanCC = sum(CCValues) / len(CCValues)
        minCC = min(CCValues)
        maxCC = max(CCValues)

        meanRMSE = sum(rmseValues) / len(rmseValues)
        minRMSE = min(rmseValues)
        maxRMSE = max(rmseValues)
        # Save final results of evaluation metrics
        saveEvalToTxt(self.config.model_name, meanMAE, minMAE, maxMAE, meanSDD, minSDD, maxSDD, meanSSIM.item(),
                      minSSIM.item(),
                      maxSSIM.item(), meanSCISSIM, minSCISSIM, maxSCISSIM, meanPSNR, minPSNR, maxPSNR, meanCC, minCC,
                      maxCC, meanRMSE, minRMSE, maxRMSE, FID_Value, time_ran, local_store_path)
        # Clean
        modelHelper.clearFolder(local_test_path.parent)


