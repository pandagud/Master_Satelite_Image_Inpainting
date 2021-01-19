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
    pathToRidgeImages = r"C:\Users\panda\PycharmProjects\Image_Inpainting_Sat\Master_Satelite_Image_Inpainting\data\processed\Belarus\T35UNB_20200617T092029\bandTCIRGB\Test\RGBImages\original_0RGB"
    images = curdatLayer.open_Imagefiles_as_array(pathToRidgeImages)

    # tuple to select colors of each channel line
    colors = ("r", "g", "b")
    channel_ids = (0, 1, 2)


    labels = []
    ridgeImages = []
    Rband = [np.arange(40000).flatten(), np.zeros((40000))]
    Gband = [np.arange(40000).flatten(), np.zeros((40000))]
    Bband = [np.arange(40000).flatten(), np.zeros((40000))]
    #Tag alle billeder, læg R kanal i en, G kanal i en, B kanal i en
    #med label Danmark_Rød, Danmark_grøn...
    #for i in range(len(images)):
    #    #for each image, put the channels with correct name into the ridgeimage and labels

    #    RGB = np.split(images[i], 3, axis=2)
    #    for j in range(3):
    #        ridgeImages.extend(RGB[j].ravel())
    #        labels.extend(np.tile(channelName[j], len(RGB[j].ravel())))
    for i in range(len(images)):
        RGB = np.split(images[i], 3, axis=2)
        uniqueR = np.unique(RGB[0], return_counts=True)
        uniqueG = np.unique(RGB[1], return_counts=True)
        uniqueB = np.unique(RGB[2], return_counts=True)
        Rband[1][uniqueR[0]] = uniqueR[1]
        Gband[1][uniqueG[0]] = uniqueG[1]
        Bband[1][uniqueB[0]] = uniqueB[1]

    #df = pd.DataFrame({'Rband': Rband[1],'Gband': Gband[1],'Bband': Bband[1]})
    dfs = []
    dfs.append(pd.DataFrame({'band':Rband[1], 'index':Rband[0], 'ChannelName': 'redBand'}))

    dfs.append(pd.DataFrame({'band':Gband[1], 'index':Gband[0], 'ChannelName': 'greenBand'}))

    dfs.append(pd.DataFrame({'band':Bband[1], 'index':Bband[0], 'ChannelName': 'blueBand'}))
    df2 = pd.concat(dfs, axis=0)
    bandsNew = []
    bandsNew.extend(Rband[1])
    bandsNew.extend(Gband[1])
    bandsNew.extend(Bband[1])
    #df = pd.DataFrame(dict(Pixel_Values=images, g=labels))
    plotting = RidgePlot().__call__(DataFrame = df2, Bands = bandsNew,Names = channelName)



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main()