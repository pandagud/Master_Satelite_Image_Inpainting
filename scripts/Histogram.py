import logging
import click
import sys
import os
import numpy as np
from src.dataLayer.importRGB import importData
from src.config_default import TrainingConfig
from src.config_utillity import update_config
from pathlib import Path
import pandas as pd
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
    curdatLayer = importData(config)

    #pathToRidgeImages = Path.joinpath(dataPath, 'Ridgeplot')
    pathToRidgeImages = r"C:\Users\panda\PycharmProjects\Image_Inpainting_Sat\Master_Satelite_Image_Inpainting\data\test\Spain\T30SUJ_20200725T110631\bandTCIRGB\Test\RGBImages\original_0RGB"
    images_test = curdatLayer.open_Imagefiles_as_array(pathToRidgeImages)
    pathToRidgeImages = r"C:\Users\panda\PycharmProjects\Image_Inpainting_Sat\Master_Satelite_Image_Inpainting\data\test\Spain\T30SUJ_20200725T110631\bandTCIRGB\Train\RGBImages\original_0RGB"
    images_train = curdatLayer.open_Imagefiles_as_array(pathToRidgeImages)
    images = images_test+images_train
    import seaborn as sns
    sns.set(font_scale=2.5)
    plt.rcParams.update({'font.size': 26})
    import cv2

    df = []

    for i in images:
        r,g,b = cv2.split(i)
        b = b.flatten()
        r = r.flatten()
        g = g.flatten()
        df.append(pd.DataFrame(np.stack([r, g, b], axis=1), columns=['Red', 'Green', 'Blue']))
    d = {'color': ['r', 'g','b']}
    df_merged = pd.concat(df)
    axes = df_merged.plot(kind='hist', subplots=True, layout=(3,1), bins=200,color=['r', 'g','b'],yticks=[], sharey=True, sharex=True)
    axes[0, 0].yaxis.set_visible(False)
    axes[1, 0].yaxis.set_visible(False)
    axes[2, 0].yaxis.set_visible(False)
    fig = axes[0, 0].figure
    fig.text(0.5, 0.04, "Pixel Value", ha="center", va="center")
    fig.text(0.05, 0.5, "Pixel frequency", ha="center", va="center", rotation=90)
    #plt.xlim(0, 4000)
    plt.show()

    STOP = True






if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main()