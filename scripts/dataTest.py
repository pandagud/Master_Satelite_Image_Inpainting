import logging
import click
import sys
import os
import numpy as np
from src.dataLayer.importRGB import importData
from src.config_default import TrainingConfig
from src.config_utillity import update_config
from src.dataLayer.importTemporalRGB import importDataTemporal
from pathlib import Path
from src.dataLayer.importRGB import importData
from src.models.train_model_new_loss import trainInpainting_test
from src.models.train_Wgan_model import trainInpaintingWgan
from src.models.UnetPartialConvModel import generator,discriminator,criticWgan,Wgangenerator
from src.config_default import TrainingConfig
from src.config_utillity import update_config
from pathlib import Path
from polyaxon_client.tracking import get_data_paths, get_outputs_path,Experiment
from src.evalMetrics.eval_GAN_model import eval_model
from src.shared.convert import _normalize,convertToFloat32
from src.shared.visualization import normalize_array_raw
import matplotlib.pyplot as plt


@click.command()
@click.argument('args', nargs=-1)
def main(args):
    """ Runs dataLayer processing scripts to turn raw dataLayer from (../raw) into
        cleaned dataLayer ready to be analyzed (saved in ../processed).
    """
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    ## Talk to Rune about how dataLayer is handle.
    config = TrainingConfig()
    config = update_config(args,config)
    ## For polyaxon
    #if config.run_polyaxon:
    localdir = Path().absolute().parent
    dataPath = Path.joinpath(localdir, 'data\ImagesForVisualization')

    logger = logging.getLogger(__name__)
    logger.info('making final dataLayer set from raw dataLayer')

    channelName = ["R", "G", "B"]
    #flatten all images used in ridgeplot so their pixel values go in one column
    path = r"C:\Users\panda\PycharmProjects\Image_Inpainting_Sat\Master_Satelite_Image_Inpainting\data\test_processed_data\Norway_winter\T32VPP_20200321T105021\bandTCIRGB\Train\RGBImages\original_0RGB"
    curdatLayer = importData(config)
    images_test = curdatLayer.open_Imagefiles_as_array(path)

    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(10, 5),
                             sharex=True, sharey=True)
    ax = axes.ravel()

    ax[0].imshow(normalize_array_raw(images_test[0]))
    ax[0].set_title('1 image')

    ax[1].imshow(normalize_array_raw(images_test[1]))
    ax[1].set_title('1 image')

    ax[2].imshow(normalize_array_raw(images_test[2]))
    ax[2].set_title('1 image')

    plt.tight_layout()
    plt.show()
    lol = ""





if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main()