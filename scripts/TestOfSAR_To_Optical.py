import logging
import click
import sys
import os
# Set PYTHONPATH to parent folder to import module.
# (https://stackoverflow.com/questions/714063/importing-modules-from-parent-folder)
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from src.dataLayer.importRGB import importData
from src.models.train_Wgan_model_SAR import trainInpaintingWgan
from src.models.UnetPartialConvModel import generator,discriminator,criticWgan
from src.config_default import TrainingConfig
from src.config_utillity import update_config
from pathlib import Path
from polyaxon_client.tracking import get_data_paths, get_outputs_path,Experiment
from src.evalMetrics.eval_GAN_model import eval_model

from src.dataLayer.LoadDataThroughRasterio import get_dataset
from src.models.UnetSarToOptical import UnetGenerator
import torch
from tqdm.auto import tqdm
from src.shared.modelUtility import modelHelper
from glob import glob

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
        input_root_path = Path(get_data_paths()['data']) #'data'
        output_root_path = Path(get_outputs_path())
        inpainting_data_path = input_root_path / 'inpainting'
        os.environ['TORCH_HOME'] = str(input_root_path / 'pytorch_cache')
        config.data_path=inpainting_data_path
        config.output_path=output_root_path
        config.polyaxon_experiment=Experiment()
        pathToData = str(input_root_path / '/workspace/data_landset8/testImages')
    else:
        pathToData = Path(r"C:\Users\Morten From\PycharmProjects\testDAta")


    logger = logging.getLogger(__name__)
    logger.info('making final dataLayer set from raw dataLayer')
    logger.info(pathToData)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    B_size = 1
    beta_test_path_list = glob(str(pathToData) + "/*/")
    ImageDict = get_dataset(beta_test_path_list, batch_size=B_size)
    train = ImageDict['train_dataloader']
    test = ImageDict['test_dataloader']

    genPath = r'C:\Users\Morten From\PycharmProjects\Speciale\Master_Satelite_Image_Inpainting\models\New_200.pth'
    outputPathImages = Path(r'C:\Users\Morten From\PycharmProjects\Speciale\Master_Satelite_Image_Inpainting\images')
    testGen = UnetGenerator(3, 3, 8)
    testGen.load_state_dict(torch.load(genPath))
    testGen = testGen.to(device)

    testGen.eval()
    iterater = 0
    for real,SAR in tqdm(train,position=0,leave=True,disable=True):
        batchOfImages = real.to(device)
        batchOfImagesSAR = SAR.to(device)
        outputs = testGen(batchOfImagesSAR)
        modelHelper.save_tensor_batchSAR(batchOfImages, batchOfImagesSAR, outputs, B_size,
                                      Path.joinpath(outputPathImages, 'iter' + str(iterater)))
        iterater = iterater+1


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main()