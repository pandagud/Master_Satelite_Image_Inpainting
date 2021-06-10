import logging
import sys
import os

# Set PYTHONPATH to parent folder to import module.
# (https://stackoverflow.com/questions/714063/importing-modules-from-parent-folder)
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from src.models.train_Wgan_model_withoutSAR import trainInpaintingWgan
from src.models.UnetPartialConvModel import generator, discriminator, criticWgan
from src.config_default import TrainingConfig
from pathlib import Path
from src.evalMetrics.eval_GAN_model import eval_model
from src.dataLayer.LoadDataThroughRasterio import get_dataset
from glob import glob


def main():
    """ Runs dataLayer processing scripts to turn raw dataLayer from (../raw) into
        cleaned dataLayer ready to be analyzed (saved in ../processed).
    """
    ## Talk to Rune about how dataLayer is handle.
    config = TrainingConfig()
    # config = update_config(args,config)
    ## For polyaxon

    config.epochs = 501
    config.run_polyaxon = True
    config.batch_size = 8
    config.lr = 0.0002
    config.save_model_step = 100
    config.n_critic = 2
    config.model_name = 'PartialConvolutionsWgan'

    # Test parametre vi kører med, som normalt sættes i experiments
    if config.run_polyaxon:
        # The POLYAXON_NO_OP env variable had to be set before any Polyaxon imports were allowed to happen
        from polyaxon import tracking
        tracking.init()
        input_root_path = Path(r'/data/inpainting/data_landset8/Test_dataset/Betaset')
        cache_path = Path('/cache')
        output_root_path = Path(tracking.get_outputs_path())
        pathToData = input_root_path ## Delete later HACK
        inpainting_data_path = input_root_path / 'inpainting'
        # Set PyTorch to use the data directory for caching pre-trained models. If this is not done, each experiment
        # will download the pre-trained model and store it in each individual experiment container, thereby wasting
        # large amounts of disk space.
        # Code is from here: https://stackoverflow.com/a/52784628
        os.environ['TORCH_HOME'] = str(cache_path / 'pytorch_cache')  # setting the environment variable

        config.output_path = Path(os.getcwd()).joinpath('outputs')
        config.data_path = Path(r'/data/inpainting/')
        config.polyaxon_tracking=tracking
    if not config.run_polyaxon:
        os.environ['POLYAXON_NO_OP'] = 'true'
    # Setup Polyaxon (import must be done here as the POLYAXON_NO_OP variable was set inside Python)


    beta_test_path_list = glob(str(pathToData) + "/*/")

    # S1A_20201005_034656_DSC_109_RGBsar_cog.tif
    # S2B_MSIL2A_20201002T090719_N0214_R050_T35TMH_20201002T113443_B02_cog
    # S2B_MSIL2A_20201002T090719_N0214_R050_T35TMH_20201002T113443_B03_cog.tif
    # S2B_MSIL2A_20201002T090719_N0214_R050_T35TMH_20201002T113443_B04_cog.tif

    logger = logging.getLogger(__name__)
    logger.info('making final dataLayer set from raw dataLayer')

    logger.info(pathToData)

    ImageDict = get_dataset(beta_test_path_list, batch_size=config.batch_size)
    train = ImageDict['train_dataloader']
    test = ImageDict['test_dataloader']

    # Kører på WGAN GP
    if config.model_name == 'PartialConvolutions':
        curtraingModel = trainInpaintingWgan(train, test, generator, criticWgan, config)
        local_model_path = curtraingModel.trainGAN()
    elif config.model_name == 'PartialConvolutionsWgan':
        curtraingModel = trainInpaintingWgan(train, test, generator, criticWgan, config)
        local_model_path = curtraingModel.trainGAN()

    # local_model_path = Path(r"C:\Users\panda\PycharmProjects\Image_Inpainting_Sat\Master_Satelite_Image_Inpainting\OutputModels\PartialConvolutionsWgan_200.pt")
    if config.run_polyaxon:
        model_path = inpainting_data_path / 'models'
        modelOutputPath = Path.joinpath(model_path, 'OutputModels')
        stores_output_path = config.output_path / 'data' / 'storedData'
    else:
        localdir = Path().absolute().parent
        modelOutputPath = Path.joinpath(localdir, 'OutputModels')
        stores_output_path = localdir / 'data' / 'storedData'


    curevalModel = eval_model(config)
    curevalModel.run_eval(modelOutputPath, stores_output_path, model_path=local_model_path, test_dataloader=test)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()