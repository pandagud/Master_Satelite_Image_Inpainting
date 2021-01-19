import logging
import click
import sys
import os
import torch
# Set PYTHONPATH to parent folder to import module.
# (https://stackoverflow.com/questions/714063/importing-modules-from-parent-folder)
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from src.dataLayer.importRGB import importData
from polyaxon_client.tracking import get_data_paths, get_outputs_path,Experiment
from pathlib import Path
from datetime import datetime
from src.dataLayer.importRGB import importData
from src.config_default import TrainingConfig
from src.config_utillity import update_config
from src.dataLayer import makeMasks
from tqdm.auto import tqdm
from polyaxon_client.tracking import get_data_paths, get_outputs_path
from src.shared.modelUtility import modelHelper
from src.models.UnetPartialConvModel import generator, Wgangenerator
from src.models.UnetPartialConvModelNIR import generatorNIR,Wgangenerator
import numpy as np
from src.evalMetrics.PSNR import PSNR
from src.evalMetrics.Pytorch_SSIM import ssim, SSIM
from src.evalMetrics.CC import CC
from src.evalMetrics.MAE import MSE
from src.evalMetrics.SDD import SDD
from src.evalMetrics.RMSE import RMSE
from src.shared.evalUtility import saveEvalToTxt
from src.evalMetrics.SSIM import SSIM_SKI
from pathlib import Path



@click.command()
@click.argument('args', nargs=-1)
def main(args):
    """ Runs dataLayer processing scripts to turn raw dataLayer from (../raw) into
        cleaned dataLayer ready to be analyzed (saved in ../processed).
    """
    ## Talk to Rune about how dataLayer is handle.
    config = TrainingConfig()
    config = update_config(args,config)
    ## For polyaxon
    if config.run_polyaxon:
        input_root_path = Path(get_data_paths()['data'])
        output_root_path = Path(get_outputs_path())
        inpainting_data_path = input_root_path / 'inpainting'
        os.environ['TORCH_HOME'] = str(input_root_path / 'pytorch_cache')
        config.data_path=inpainting_data_path
        config.output_path=output_root_path
        config.polyaxon_experiment=Experiment()

    logger = logging.getLogger(__name__)
    logger.info('making final dataLayer set from raw dataLayer')
    curdatLayer = importData(config)

    mask_path= r"C:\Users\panda\PycharmProjects\Image_Inpainting_Sat\Master_Satelite_Image_Inpainting\data\mask\experiment_mask_dataset"
    images_test,names = curdatLayer.open_bandImage_as_array_and_names(mask_path)
    for index,i in enumerate(images_test):
        mask_size= np.count_nonzero(i)
        percent_cover = mask_size/i.size*100.0
        print("The mask"+str(names[index]+" had a percent cover at "+ str(percent_cover)))

    train, test_dataloader = curdatLayer.getRGBDataLoader()
    local_model_path= r"C:\Users\panda\PycharmProjects\Image_Inpainting_Sat\Master_Satelite_Image_Inpainting\OutputModels\PartialConvolutionsWgan_901.pt"
    local_output_path =Path(r"E:\Speciale\MaskExperiment")
    #gen = Wgangenerator().to(config.device)
    gen = generator().to(config.device)
    gen.load_state_dict(torch.load(local_model_path))  ## Use epochs to identify model number
    gen.eval()

    loadAndAgumentMasks = makeMasks.MaskClass(config, rand_seed=None, evaluation=True,noFlip=True,experiment=True)
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
        if config.run_polyaxon:
            selected_image = i.split('/')[-1]  ##Linux
        else:
            selected_image = i.split("\\")[-1]
        localNames.append(selected_image)
    names = names + localNames

    print("Found this many names " + str(len(names)))

    current_number = 0

    if not os.path.exists(Path.joinpath(local_output_path, config.model_name)):
        os.makedirs(Path.joinpath(local_output_path, config.model_name))

    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")

    local_test_path = local_output_path / config.model_name / dt_string / 'Data'
    local_test_nir_path = local_output_path / config.model_name / dt_string / 'DataNir'
    local_store_path = local_output_path / config.model_name / dt_string / 'generated_Data'
    os.makedirs(local_test_path)
    os.makedirs(local_store_path)
    start_time = datetime.now()
    number_of_masks= 7
    for real in tqdm(test_dataloader, disable=config.run_polyaxon):
        for i in range(number_of_masks):
            masks = loadAndAgumentMasks.returnTensorMasks(config.batch_size, i)
            masks = torch.from_numpy(masks)
            masks = masks.type(torch.cuda.FloatTensor)
            masks = 1 - masks
            masks.to(config.device)

            real = real.to(config.device)
            fake_masked_images = torch.mul(real, masks)
            generated_images = gen(fake_masked_images, masks)
            image_names = names[current_number:current_number + config.batch_size]
            current_number = current_number + config.batch_size  ## Change naming to include all names
            # modelHelper.save_tensor_batch(generated_images,fake_masked_images,config.batch_size,path)
            modelHelper.save_tensor_batch(real, fake_masked_images, generated_images, config.batch_size,
                                          Path.joinpath(local_test_path, "Inpainted_mask_" + str(image_names)))

            maeValues = []
            sddValues = []
            ssimscikitValues = []
            SSIMValues = []
            psnrValues = []
            CCValues = []
            rmseValues = []
            real_eval = torch.squeeze(real)
            generated_images_eval=torch.squeeze(generated_images)
            psnrValues.append(PSNR().__call__(real_eval, generated_images_eval))
            CCValues.append(CC().__call__(real_eval, generated_images_eval))
            maeValues.append(MSE().__call__(real_eval, generated_images_eval))
            sddValues.append(SDD.__call__(real_eval, generated_images_eval))
            ssimscikitValues.append(SSIM_SKI.__call__(real_eval, generated_images_eval))
            rmseValues.append(RMSE.__call__(real_eval, generated_images_eval))
            meanMAE = sum(maeValues) / len(maeValues)
            minMAE = min(maeValues)
            maxMAE = max(maeValues)

            meanSDD = sum(sddValues) / len(sddValues)
            minSDD = min(sddValues)
            maxSDD = max(sddValues)

            meanPSNR = sum(psnrValues) / len(psnrValues)
            minPSNR = min(psnrValues)
            maxPSNR = max(psnrValues)

            meanSSIM = i
            minSSIM = i
            maxSSIM = i

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
            saveEvalToTxt(config.model_name, meanMAE, minMAE, maxMAE, meanSDD, minSDD, maxSDD, meanSSIM,
                          minSSIM,
                          maxSSIM, meanSCISSIM, minSCISSIM, maxSCISSIM, meanPSNR, minPSNR, maxPSNR, meanCC,
                          minCC, maxCC, meanRMSE, minRMSE, maxRMSE, 0, 0, local_output_path)
        break

    end_time = datetime.now()


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main()