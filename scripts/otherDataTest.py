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
    path = Path(r"C:\Users\panda\OneDrive\Skrivebord\ComputerTekonlogi\Speciale\OtherDataset\singleImage\clear")
    import glob
    import cv2
    str_path = str(path)
    filelist = glob.glob(str_path + '/*.jpg')
    data = []
    for fname in filelist:
        image = cv2.imread(fname, -1)
        data.append(image)
    images = data
    import seaborn as sns
    sns.set(font_scale=2.5)
    plt.rcParams.update({'font.size': 26})
    print("Max " +str(np.max(images)))
    print("Min "+str(np.min(images)))
    print("Std " +str(np.std(images)))
    print("Mean " +str(np.mean(images)))
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
    plt.xlim(0, 255)
    plt.show()

    STOP = True






if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main()