import logging
import click
from pathlib import Path
from src.dataLayer.importRGB import importData
from src.models.train_model import trainInpainting
from src.models.train_Wgan_model import trainInpaintingWgan
from src.models.UnetPartialConvModel import generator,discriminator,criticWgan
from src.config_default import TrainingConfig
from src.config_utillity import update_config


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
    train, test = curdatLayer.getRGBDataLoader()
    if config.model_name == 'PartialConvolutions':
        curtraingModel=trainInpainting(train,test,generator,discriminator,config)
        curtraingModel.trainGAN()
    elif config.model_name == 'ParTialConvolutionsWgan':
        curtraingModel = trainInpaintingWgan(train, test, generator, criticWgan, config)
        curtraingModel.trainGAN()



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main()