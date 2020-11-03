import logging
import click
from pathlib import Path
from src.dataLayer.importRGB import importData
from src.models.train_model import trainInpainting
from src.models.baseline_Model import baselineModel
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

    import matplotlib.image as mpimg
    import numpy as np
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = 1000000000

    # print('Reading B04.jp2...')
    img_red = mpimg.imread(r"C:\Users\panda\PycharmProjects\Image_Inpainting_Sat\Master_Satelite_Image_Inpainting\data\processed\T32UPV_20190904T102021\bandTCIRGB\Train\redBand\_01_01.tiff")

    print('Reading B03.jp2...')
    img_green = mpimg.imread(r"C:\Users\panda\PycharmProjects\Image_Inpainting_Sat\Master_Satelite_Image_Inpainting\data\processed\T32UPV_20190904T102021\bandTCIRGB\Train\greenBand\_01_01.tiff")

    print('Reading B02.jp2...')
    img_blue = mpimg.imread(r"C:\Users\panda\PycharmProjects\Image_Inpainting_Sat\Master_Satelite_Image_Inpainting\data\processed\T32UPV_20190904T102021\bandTCIRGB\Train\blueBand\_01_01.tiff")

    img = np.dstack((img_red, img_green, img_blue))
    #img = mpimg.imread(r"C:\Users\panda\PycharmProjects\Image_Inpainting_Sat\Master_Satelite_Image_Inpainting\scripts\real.tiff")
    #img = (img / np.iinfo(img.dtype).max)
    img = img * 255.0 / img.max()
    img = img.astype(np.uint8)

    mpimg.imsave('MIX2.jpeg', img, format='jpg')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main()



