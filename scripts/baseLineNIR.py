import logging
import click
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

    curdatLayer = importData(config)
    train_array, names = curdatLayer.get_images_for_baseLine()
    print("Total test in baseline " + str(len(train_array)))
    print("Total test names in baseline" + str(len(names)))
    train_nir,names = curdatLayer.get_images_for_baseLine_NIR()
    print("Total test in baseline " + str(len(train_nir)))
    print("Total test names in baseline" + str(len(names)))
    train_dataloader, test_dataloader = curdatLayer.getRGBDataLoader()
    local_train_array = []
    local_train_nir_array=[]
    for i in train_array:
        local_train_array.append(convertToFloat32(i))
    for i in train_nir:
        local_train_nir_array.append(convertToFloat32(i))
    train_array = local_train_array
    train_nir = local_train_nir_array
    curBaseLineModel = baselineModel(train_array, names, config)
    pathToGenerated, time_ran = curBaseLineModel.baselineExperimentNIR(train_nir)
    # pathToGenerated = r"C:\Users\panda\PycharmProjects\Image_Inpainting_Sat\Master_Satelite_Image_Inpainting\data\generated\test_baseLine\22_11_2020_13_01_28"



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main()



