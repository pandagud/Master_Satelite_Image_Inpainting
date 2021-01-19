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
from src.evalMetrics.SSIM import SSIM_SKI
from src.shared.convert import convertToFloat32
import cv2
from src.evalMetrics.PSNR import PSNR
from src.shared.evalUtility import saveEvalToTxt
from src.evalMetrics.Pytorch_SSIM import ssim, SSIM
from src.evalMetrics.CC import CC
from src.evalMetrics.MAE import MSE
from src.evalMetrics.SDD import SDD
from src.evalMetrics.RMSE import RMSE
from src.evalMetrics.SSIM import SSIM_SKI
from src.dataLayer import  makeMasks
import earthpy.plot as ep

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

    curdatLayer = importData(config)

    ## Original
    pathToNIR = r"E:\Speciale\NDVIExperiment\Croatia\Original_Data\NIR"
    nir_images = curdatLayer.open_Imagefiles_as_array(pathToNIR)
    nir_image = nir_images[0]
    pathtoRGB = r"E:\Speciale\NDVIExperiment\Croatia\Original_Data\RGB"
    rgb_images=curdatLayer.open_Imagefiles_as_array(pathtoRGB)
    rgb_image= rgb_images[0]
    r,g,b = cv2.split(rgb_image)

    org_ndvi = (nir_image - r) / (nir_image + r)
    titles = ["Sentinel 2 - Normalized Difference Vegetation Index (NDVI) over Original"]
    # https://earthpy.readthedocs.io/en/latest/gallery_vignettes/plot_calculate_classify_ndvi.html
    # Turn off bytescale scaling due to float values for NDVI
    ep.plot_bands(org_ndvi, cmap="RdYlGn", cols=1, title=titles, vmin=-1, vmax=1)

    ## Inpainted
    pathToNIR = r"E:\Speciale\NDVIExperiment\Croatia\PartialConvolutions\big_mask\DataNir"
    nir_images = curdatLayer.open_Imagefiles_as_array(pathToNIR)
    nir_image = nir_images[0]
    pathtoRGB = r"E:\Speciale\NDVIExperiment\Croatia\PartialConvolutions\big_mask\Data"
    rgb_images = curdatLayer.open_Imagefiles_as_array(pathtoRGB)
    rgb_image = rgb_images[0]
    r, g, b = cv2.split(rgb_image)

    gen_ndvi = (nir_image - r) / (nir_image + r)
    titles = ["Sentinel 2- Normalized Difference Vegetation Index (NDVI) over generated"]
    # https://earthpy.readthedocs.io/en/latest/gallery_vignettes/plot_calculate_classify_ndvi.html
    # Turn off bytescale scaling due to float values for NDVI
    ep.plot_bands(gen_ndvi, cmap="RdYlGn", cols=1, title=titles, vmin=-1, vmax=1)

    diff_ndvi=org_ndvi-gen_ndvi
    old= sum(gen_ndvi.flatten())
    new = sum(org_ndvi.flatten())
    diffSumsWithMaria = ((new-old)/old)
    diff_percent_sum = sum((gen_ndvi.flatten()-org_ndvi.flatten())/org_ndvi.flatten()*100)

    print("The NDVI have changed " +str(diffSumsWithMaria)+" %")

    titles = ["Sentinel 2- Normalized Difference Vegetation Index (NDVI) difference"]
    # https://earthpy.readthedocs.io/en/latest/gallery_vignettes/plot_calculate_classify_ndvi.html
    # Turn off bytescale scaling due to float values for NDVI
    ep.plot_bands(diff_ndvi, cmap="RdYlGn", cols=1, title=titles, vmin=-1, vmax=1)
    loadAndAgumentMasks = makeMasks.MaskClass(config, rand_seed=None, evaluation=True)
    mask = loadAndAgumentMasks.returnMask(787)
    mask = mask[0, :, :]
    # Get real and set to GPU
    #Invert
    mask = 1-mask
    # Augment with masks
    # Check if this applies to  all three color channels?
    gen_ndvi_masked = gen_ndvi.copy()
    org_ndvi_masked = org_ndvi.copy()
    for layer in range(gen_ndvi_masked.shape[-1]):
        gen_ndvi_masked[np.where(mask)] = 0
    for layer in range(org_ndvi_masked.shape[-1]):
        org_ndvi_masked[np.where(mask)] = 0

    ep.plot_bands(gen_ndvi_masked, cmap="RdYlGn", cols=1, title=titles, vmin=-1, vmax=1)
    maeValues = []
    sddValues = []
    ssimscikitValues = []
    psnrValues = []
    CCValues = []
    rmseValues = []
    org_ndvi_masked=org_ndvi_masked[org_ndvi_masked!=0]
    gen_ndvi_masked=gen_ndvi_masked[gen_ndvi_masked!=0]
    psnrValues.append(PSNR().__call__(org_ndvi_masked, gen_ndvi_masked,tensor=False))
    CCValues.append(CC().__call__(org_ndvi_masked, gen_ndvi_masked,tensor=False))
    maeValues.append(MSE().__call__(org_ndvi_masked, gen_ndvi_masked,tensor=False))
    sddValues.append(SDD.__call__(org_ndvi_masked, gen_ndvi_masked,tensor=False))
    #ssimscikitValues.append(SSIM_SKI.__call__(org_ndvi_masked, gen_ndvi_masked,tensor=False))
    rmseValues.append(RMSE.__call__(org_ndvi_masked, gen_ndvi_masked,tensor=False))
    meanMAE = sum(maeValues) / len(maeValues)
    minMAE = min(maeValues)
    maxMAE = max(maeValues)

    meanSDD = sum(sddValues) / len(sddValues)
    minSDD = min(sddValues)
    maxSDD = max(sddValues)

    meanPSNR = sum(psnrValues) / len(psnrValues)
    minPSNR = min(psnrValues)
    maxPSNR = max(psnrValues)

    meanSSIM = 0
    minSSIM = 0
    maxSSIM = 0

    meanSCISSIM = 0
    minSCISSIM = 0
    maxSCISSIM = 0
    # meanSCISSIM = sum(ssimscikitValues) / len(ssimscikitValues)
    # minSCISSIM = min(ssimscikitValues)
    # maxSCISSIM = max(ssimscikitValues)

    meanCC = sum(CCValues) / len(CCValues)
    minCC = min(CCValues)
    maxCC = max(CCValues)

    meanRMSE = sum(rmseValues) / len(rmseValues)
    minRMSE = min(rmseValues)
    maxRMSE = max(rmseValues)
    FID_Value=0.0
    time_ran=0.0
    local_store_path=Path(r"E:\Speciale\NDVIExperiment\Croatia")
    saveEvalToTxt(config.model_name, meanMAE, minMAE, maxMAE, meanSDD, minSDD, maxSDD, meanSSIM,
                  minSSIM,
                  maxSSIM, meanSCISSIM, minSCISSIM, maxSCISSIM, meanPSNR, minPSNR, maxPSNR, meanCC, minCC, maxCC,
                  meanRMSE, minRMSE, maxRMSE, FID_Value, time_ran, local_store_path)

    # v_nir_image = np.concatenate((nir_images[0], nir_images[1]), axis=1)
    # v_rgb_image = np.concatenate((rgb_images[0], rgb_images[1]), axis=1)
    # v_r, v_g, v_b = cv2.split(v_rgb_image)
    #
    # v_ndvi = (v_nir_image - v_r) / (v_nir_image + v_r)
    # titles = ["Sentinel 2- Normalized Difference Vegetation Index (NDVI) over two samples"]
    # # https://earthpy.readthedocs.io/en/latest/gallery_vignettes/plot_calculate_classify_ndvi.html
    # # Turn off bytescale scaling due to float values for NDVI
    # ep.plot_bands(v_ndvi, cmap="RdYlGn", cols=1, title=titles, vmin=-1, vmax=1)
    # nir_images= nir_images/10000
    # r = r/10000
    # ndvi = (nir_images - r) / (nir_images + r)
    #
    # fig = plt.figure(figsize=(10, 10))
    # fig.set_facecolor('white')
    # plt.imshow(ndvi, cmap='RdYlGn')  # Typically the color map for NDVI maps are the Red to Yellow to Green
    # plt.title('NDVI')
    # plt.show()
    #



    lol = ""
if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main()