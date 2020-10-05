import logging
import click
from pathlib import Path
from src.dataLayer.importRGB import importData
from src.models.train_model import trainInpainting
from src.models.UnetPartialConvModel import generator,discriminator


@click.command()
@click.argument('input_filepath', type=click.Path())
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs dataLayer processing scripts to turn raw dataLayer from (../raw) into
        cleaned dataLayer ready to be analyzed (saved in ../processed).
    """
    ## Talk to Rune about how dataLayer is handle. If it should be part of the "big" project.

    logger = logging.getLogger(__name__)
    logger.info('making final dataLayer set from raw dataLayer')

    print("the argument(s) " +input_filepath)
    print("the argument(s) " + output_filepath)
    curdatLayer = importData()
    train, test = curdatLayer.getRBGDataLoader()
    gen = generator()
    disc = discriminator()
    curtraingModel=trainInpainting(train,test,gen,disc)
    curtraingModel.trainGAN()

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main()