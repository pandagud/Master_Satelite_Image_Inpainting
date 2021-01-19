import logging
import click
import sys
import os
import numpy as np
# Set PYTHONPATH to parent folder to import module.
# (https://stackoverflow.com/questions/714063/importing-modules-from-parent-folder)
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from pathlib import Path
from src.dataLayer.importRGB import importData
from src.models.baseline_Model import baselineModel
from src.config_default import TrainingConfig
from src.config_utillity import update_config
from src.evalMetrics.FID import FIDCalculator
from src.evalMetrics.PSNR import PSNR
from src.shared.evalUtility import saveEvalToTxt
from src.evalMetrics.Pytorch_SSIM import ssim, SSIM
from src.evalMetrics.CC import CC
from src.evalMetrics.MAE import MSE
from src.evalMetrics.SDD import SDD
from src.evalMetrics.RMSE import RMSE
from src.evalMetrics.SSIM import SSIM_SKI
from pathlib import Path
from polyaxon_client.tracking import get_data_paths, get_outputs_path,Experiment
from src.shared.convert import convertToFloat32


@click.command()
@click.argument('args', nargs=-1)
def main(args):
    """ Runs dataLayer processing scripts to turn raw dataLayer from (../raw) into
        cleaned dataLayer ready to be analyzed (saved in ../processed).
    """
    ## Talk to Rune about how dataLayer is handle.
    config = TrainingConfig()
    config = update_config(args,config)
    logger = logging.getLogger(__name__)
    logger.info('making final dataLayer set from raw dataLayer')

    ## For polyaxon
    if config.run_polyaxon:
        input_root_path = Path(get_data_paths()['data'])
        output_root_path = Path(get_outputs_path())
        inpainting_data_path = input_root_path / 'inpainting'
        os.environ['TORCH_HOME'] = str(input_root_path / 'pytorch_cache')
        config.data_path = inpainting_data_path
        config.output_path = output_root_path
        config.polyaxon_experiment = Experiment()

    curdatLayer = importData(config)
    train_array,names = curdatLayer.get_images_for_baseLine()
    print("Total test in baseline " +str(len(train_array)))
    print("Total test names in baseline" +str(len(names)))
    train_dataloader,test_dataloader = curdatLayer.getRGBDataLoader()
    local_train_array = []
    for i in train_array:
        local_train_array.append(convertToFloat32(i))
    train_array = local_train_array
    curBaseLineModel = baselineModel(train_array,names,config)
    pathToGenerated, time_ran = curBaseLineModel.baselineExperiment()
    #pathToGenerated = r"C:\Users\panda\PycharmProjects\Image_Inpainting_Sat\Master_Satelite_Image_Inpainting\data\generated\test_baseLine\22_11_2020_13_01_28"
    if config.run_polyaxon:
        pathToEval=config.output_path /'evalMetrics'
    else:
        pathToEval = Path().absolute().parent / 'models'
    # create dataloader with generated images
    generated_images_dataloader = curdatLayer.getGeneratedImagesDataloader(pathToGenerated)
    print(str(pathToGenerated) + "is the generated image path")
    # calculate FID
    #missing gen
    print("Done with generating images")
    FID_Value = FIDCalculator(test_dataloader, generated_images_dataloader,
                              len(test_dataloader) * config.batch_size, config.batch_size,config).get_FID_scores()

    # Calculate PSNR and SSIM
    dataloader_iterator = iter(generated_images_dataloader)
    maeValues = []
    sddValues = []
    ssimscikitValues= []
    SSIMValues = []
    psnrValues = []
    CCValues = []
    rmseValues = []
    # loop to calculate PSNR and SSIM for all test and generated images.

    for images_real in test_dataloader:
        try:
            images_generated  = next(dataloader_iterator)
        except StopIteration:
            dataloader_iterator  = iter(generated_images_dataloader)
            images_generated = next(dataloader_iterator)
        for index2 in range(config.batch_size):
            psnrValues.append(PSNR().__call__(images_real[index2], images_generated[index2]))
            CCValues.append(CC().__call__(images_real[index2], images_generated[index2]))
            maeValues.append(MSE().__call__(images_real[index2], images_generated[index2]))
            sddValues.append(SDD.__call__(images_real[index2], images_generated[index2]))
            ssimscikitValues.append(SSIM_SKI.__call__(images_real[index2], images_generated[index2]))
            image1 = images_real[index2].unsqueeze(0)
            image2 = images_generated[index2].unsqueeze(0)
            SSIMValues.append(ssim(image1, image2))
            rmseValues.append(RMSE.__call__(images_real[index2], images_generated[index2]))



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
    FID = FID_Value
    if not pathToEval.parent.exists():
        pathToEval.parent.mkdir()
    #saveEvalToTxt(config.model_name,meanPSNR.item(),minPSNR,maxPSNR,meanSSIM.item(), minSSIM,maxSSIM ,FID ,time, pathToEval)
    saveEvalToTxt(config.model_name, meanMAE, minMAE, maxMAE, meanSDD, minSDD, maxSDD, meanSSIM.item(),
                  minSSIM.item(),
                  maxSSIM.item(),meanSCISSIM,minSCISSIM,maxSCISSIM, meanPSNR, minPSNR, maxPSNR, meanCC, minCC, maxCC, meanRMSE, minRMSE, maxRMSE,
                  FID_Value, time_ran, pathToEval)
    #modelHelper.clearFolder(pathToGenerated)
    #modelHelper.clearFolder(pathToGenerated.parent)
    #modelHelper.clearFolder(pathToGenerated.parent.parent)
    #saveEvalToTxt(config.model_name,meanPSNR.item(),minPSNR,maxPSNR,meanSSIM.item(), minSSIM,maxSSIM ,FID ,time, pathToEval)
    # modelHelper.clearFolder(pathToGenerated)
    # modelHelper.clearFolder(pathToGenerated.parent)
    # modelHelper.clearFolder(pathToGenerated.parent.parent)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main()



