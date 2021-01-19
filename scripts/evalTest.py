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
from src.shared.convert import convertToFloat32,_normalize
from src.evalMetrics.eval_helper import remove_outliers
import numpy as np
import matplotlib.pyplot as plt

from skimage import data, img_as_float
from skimage.metrics import structural_similarity as ssim
from src.shared.visualization import normalize_array_raw,normalize_array
from skimage.restoration import inpaint
from skimage import img_as_float32
from skimage.measure import compare_psnr, compare_ssim
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

    def mse(x, y):
        return np.linalg.norm(x - y)

    def MaxMin(a, b):
        minvalue = min(a.min(), b.min())
        maxvalue = max(a.max(), b.max())
        return maxvalue - minvalue
    logger = logging.getLogger(__name__)
    logger.info('making final dataLayer set from raw dataLayer')
    plt.rcParams.update({'font.size': 18})
    path_generated = r"C:\Users\panda\PycharmProjects\Image_Inpainting_Sat\Master_Satelite_Image_Inpainting\data\generated\Base_Line\18_12_2020_09_58_50\Data"
    curdatLayer = importData(config)
    generated_images = curdatLayer.open_Imagefiles_as_array(path_generated)
    generated_img = generated_images[0]
    path_real = r"C:\Users\panda\PycharmProjects\Image_Inpainting_Sat\Master_Satelite_Image_Inpainting\data\processed\Belarus\T35UNB_20200617T092029\bandTCIRGB\Test\RGBImages\original_0RGB"
    original_images = curdatLayer.open_Imagefiles_as_array(path_real)
    org_img = original_images[0]
    another_img=original_images[1]
    win_size=11

    generated_img = convertToFloat32(generated_img)
    org_img = convertToFloat32(org_img)
    another_img = convertToFloat32(another_img)
    generated_img_copy = generated_img.copy()
    # generated_img = remove_outliers(generated_img)
    org_img_copy = org_img.copy()
    # org_img = remove_outliers(org_img)
    another_img_copy=another_img.copy()
    # another_img = remove_outliers(another_img)
    #
    mse_org = mse(org_img, org_img)
    ssim_org, ssim_org_full = ssim(org_img, org_img, data_range=MaxMin(org_img,org_img),
                    multichannel=True, win_size=win_size,full=True)

    mse_org_vs_inpaint = mse(org_img, generated_img)
    ssim_org_vs_inpaint,ssim_org_vs_inpaint_full = ssim(org_img, generated_img, data_range=MaxMin(org_img,generated_img),
                               multichannel=True, win_size=win_size,full=True)

    mse_org_vs_another = mse(org_img, another_img)
    ssim_org_vs_another,ssim_org_vs_another_full = ssim(org_img, another_img, data_range=MaxMin(org_img,another_img),
                               multichannel=True, win_size=win_size,full=True)


    fig, axes = plt.subplots(nrows=2, ncols=3,
                             sharex=True, sharey=True)
    ax = axes.ravel()
    label = 'MSE: {:.2f}, SSIM: {:.2f}'

    ax[0].imshow(_normalize(org_img_copy))
    ax[0].set_xlabel(label.format(mse_org, ssim_org))
    ax[0].set_title('Original image')
    ax[0].set_yticklabels([])
    ax[0].set_xticklabels([])

    ax[1].imshow(_normalize(generated_img_copy))
    ax[1].set_xlabel(label.format(mse_org_vs_inpaint, ssim_org_vs_inpaint))
    ax[1].set_title('Original vs inpainted')
    ax[1].set_yticklabels([])
    ax[1].set_xticklabels([])

    ax[2].imshow(_normalize(another_img_copy))
    ax[2].set_xlabel(label.format(mse_org_vs_another, ssim_org_vs_another))
    ax[2].set_title('Original vs Another Image')
    ax[2].set_yticklabels([])
    ax[2].set_xticklabels([])

    ax[3].imshow(ssim_org_full)
    ax[3].set_title('Original image')
    ax[3].set_yticklabels([])
    ax[3].set_xticklabels([])

    ax[4].imshow(ssim_org_vs_inpaint_full)
    ax[4].set_title('Original vs inpainted')
    ax[4].set_yticklabels([])
    ax[4].set_xticklabels([])

    ax[5].imshow(ssim_org_vs_another_full)
    ax[5].set_title('Original vs Another Image')
    ax[5].set_yticklabels([])
    ax[5].set_xticklabels([])



    plt.tight_layout()
    plt.show()
    #
    #
    noise = np.ones_like(org_img) * 0.2 * (org_img.max() - org_img.min())
    # noise[np.random.random(size=noise.shape) > 0.5] *= -1
    #
    #
    #
    # img_noise = org_img + noise
    # img_const = org_img + abs(noise)
    #
    # win_size = 11
    #
    # #image_result = convertToFloat32(image_result)
    # #org_img_float = convertToFloat32(org_img_float)
    # #image_defect = convertToFloat32(image_defect)
    #
    #
    # mse_none = mse(org_img, org_img)
    # ssim_none,ssim_none_full = ssim(org_img, org_img,multichannel=True , data_range=MaxMin(org_img,org_img)
    #                                 ,gaussian_weights=True,win_size=win_size,full=True)
    # # ssim_none = compare_ssim(
    # #     generated_img,
    # #     generated_img,
    # #     win_size=11,
    # #     gaussian_weights=True,
    # #     multichannel=True,
    # #     data_range=1.0,
    # #     K1=0.01,
    # #     K2=0.03,
    # #     sigma=1.5)
    # mse_noise = mse(org_img, img_noise)
    # ssim_noise,ssim_noise_full = ssim(org_img, img_noise,multichannel=True , data_range=MaxMin(org_img,img_noise)
    #                                   , gaussian_weights=True,win_size=win_size,full=True)
    # # ssim_noise = compare_ssim(
    # #     generated_img,
    # #     img_noise,
    # #     win_size=11,
    # #     gaussian_weights=True,
    # #     multichannel=True,
    # #     data_range=1.0,
    # #     K1=0.01,
    # #     K2=0.03,
    # #     sigma=1.5)
    # mse_const = mse(org_img, img_const)
    # ssim_const,ssim_const_full = ssim(org_img, img_const,multichannel=True, data_range=MaxMin(org_img,img_const)
    #                                   ,gaussian_weights=True,win_size=win_size,full=True)
    #
    # fig, axes = plt.subplots(nrows=2, ncols=3,
    #                          sharex=True, sharey=True)
    # ax = axes.ravel()
    # label = 'MSE: {:.2f}, SSIM: {:.2f}'
    #
    # ax[0].imshow(_normalize(org_img))
    # ax[0].set_xlabel(label.format(mse_none, ssim_none))
    # ax[0].set_title('Original image')
    # ax[0].set_yticklabels([])
    # ax[0].set_xticklabels([])
    #
    # ax[1].imshow(_normalize(img_noise))
    # ax[1].set_xlabel(label.format(mse_noise, ssim_noise))
    # ax[1].set_title('Original vs noise')
    # ax[1].set_yticklabels([])
    # ax[1].set_xticklabels([])
    #
    # ax[2].imshow(_normalize(img_const))
    # ax[2].set_xlabel(label.format(mse_const, ssim_const))
    # ax[2].set_title('Original vs constant')
    # ax[2].set_yticklabels([])
    # ax[2].set_xticklabels([])
    #
    # ax[3].imshow(ssim_none_full)
    # ax[3].set_title('Original image')
    # ax[3].set_yticklabels([])
    # ax[3].set_xticklabels([])
    #
    # ax[4].imshow(ssim_noise_full)
    # ax[4].set_title('Original vs inpainted')
    # ax[4].set_yticklabels([])
    # ax[4].set_xticklabels([])
    #
    # ax[5].imshow(ssim_const_full)
    # ax[5].set_title('Original vs noneinpaited')
    # ax[5].set_yticklabels([])
    # ax[5].set_xticklabels([])
    #
    # plt.tight_layout()
    # plt.show()
    #
    #
    #
    #
    #
    #
    #
    #
    norm = np.linalg.norm(org_img)
    # org_img_copy = org_img/norm
    # generated_img_copy = generated_img/norm
    # mse_generated_vs_original = mse(generated_img_copy,org_img_copy)
    # ssim_generated_vs_original, ssim_image = ssim(generated_img_copy, org_img_copy,
    #                  data_range=MaxMin(generated_img_copy,org_img_copy),multichannel=True,gaussian_weights=True,win_size=win_size,full=True)
    # # ssim_generated_vs_original = compare_ssim(
    # #     generated_img,
    # #     org_img,
    # #     win_size=11,
    # #     gaussian_weights=True,
    # #     multichannel=True,
    # #     data_range=1.0,
    # #     K1=0.01,
    # #     K2=0.03,
    # #     sigma=1.5)
    # mse_anotherimg_vs_generated= mse(generated_img,another_img)
    # ssim_anotherimg_vs_generated=ssim(generated_img,another_img,data_range=MaxMin(generated_img,another_img),multichannel=True,gaussian_weights=True,win_size=win_size)
    # # ssim_anotherimg_vs_generated= compare_ssim(
    # #     generated_img,
    # #     another_img,
    # #     win_size=11,
    # #     gaussian_weights=True,
    # #     multichannel=True,
    # #     data_range=1.0,
    # #     K1=0.01,
    # #     K2=0.03,
    # #     sigma=1.5)
    #
    # mse_new_mask = mse(org_img, image_result)
    # ssim_new_mask= ssim(org_img, image_result, data_range=MaxMin(org_img_float,image_result),
    #                                     multichannel=True,win_size=win_size)
    # # ssim_new_mask = compare_ssim(
    # #     org_img_float,
    # #     image_result,
    # #     win_size=11,
    # #     gaussian_weights=True,
    # #     multichannel=True,
    # #     data_range=1.0,
    # #     K1=0.01,
    # #     K2=0.03,
    # #     sigma=1.5)
    # mse_with_mask = mse(org_img,image_defect)
    # ssim_with_mask= ssim(org_img, image_defect, data_range=MaxMin(org_img_float,image_defect),
    #                      multichannel=True, win_size=win_size)
    # # ssim_with_mask = compare_ssim(
    # #     org_img_float,
    # #     image_defect,
    # #     win_size=11,
    # #     gaussian_weights=True,
    # #     multichannel=True,
    # #     data_range=1.0,
    # #     K1=0.01,
    # #     K2=0.03,
    # #     sigma=1.5)
    #
    # fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(10, 5),
    #                          sharex=True, sharey=True)
    # ax = axes.ravel()
    # label = 'MSE: {:.2f}, SSIM: {:.2f}'
    #
    # ax[0].imshow(_normalize(generated_img))
    # ax[0].set_xlabel(label.format(mse_none, ssim_none))
    # ax[0].set_title('Generated image')
    #
    # ax[1].imshow(_normalize(img_noise))
    # ax[1].set_xlabel(label.format(mse_noise, ssim_noise))
    # ax[1].set_title('Image with noise')
    #
    # ax[2].imshow(_normalize(img_const))
    # ax[2].set_xlabel(label.format(mse_const, ssim_const))
    # ax[2].set_title('Image plus constant')
    #
    # ax[3].imshow(_normalize(org_img))
    # ax[3].set_xlabel(label.format(mse_generated_vs_original, ssim_generated_vs_original))
    # ax[3].set_title('Original Image')
    #
    # ax[4].imshow(_normalize(another_img))
    # ax[4].set_xlabel(label.format(mse_anotherimg_vs_generated, ssim_anotherimg_vs_generated))
    # ax[4].set_title('A different Image')
    #
    # ax[5].imshow(_normalize(image_result))
    # ax[5].set_xlabel(label.format(mse_new_mask, ssim_new_mask))
    # ax[5].set_title('Original vs Generated with different Mask')
    #
    # ax[6].imshow(_normalize(image_defect))
    # ax[6].set_xlabel(label.format(mse_with_mask, ssim_with_mask))
    # ax[6].set_title('Defected image with no inpainting on')
    #
    # ax[7].imshow(ssim_image)
    # ax[7].set_title('SSIM image')
    #
    # plt.tight_layout()
    # plt.show()
    win_size = 11

    def mse(x, y):
        return np.linalg.norm(x - y)

    def MaxMin(a, b):
        minvalue = min(a.min(),b.min())
        maxvalue = max(a.max(),b.max())
        return maxvalue-minvalue

    image_orig = data.astronaut()[0:200, 0:200]

    # Create mask with three defect regions: left, middle, right respectively
    mask = np.zeros(image_orig.shape[:-1])
    mask[20:60, 0:20] = 1
    mask[160:180, 70:155] = 1
    mask[30:60, 170:195] = 1

    # Defect image over the same region in each color channel
    image_defect = image_orig.copy()
    for layer in range(image_defect.shape[-1]):
        image_defect[np.where(mask)] = 0

    image_result = inpaint.inpaint_biharmonic(image_defect, mask,
                                              multichannel=True)
    image_orig = img_as_float(image_orig)
    image_defect= img_as_float(image_defect)
    mse_org = mse(image_orig, image_orig)
    ssim_org,ssim_org_full= ssim(image_orig, image_orig, data_range=MaxMin(image_orig,image_orig),
                               multichannel=True, win_size=win_size, full=True)

    mse_org_vs_inpaint = mse(image_orig, image_result)
    ssim_org_vs_inpaint,ssim_org_vs_inpaint_full = ssim(image_orig, image_result,   data_range=MaxMin(image_orig,image_result),
                               multichannel=True, win_size=win_size,full=True)

    mse_org_vs_masked= mse(image_orig, image_defect)
    ssim_org_vs_masked,ssim_org_vs_masked_full = ssim(image_orig, image_defect, data_range=MaxMin(image_orig,image_defect),
                               multichannel=True, win_size=win_size,full=True)

    fig, axes = plt.subplots(nrows=2, ncols=3,
                             sharex=True, sharey=True)
    ax = axes.ravel()
    label = 'MSE: {:.2f}, SSIM: {:.2f}'

    ax[0].imshow(image_orig)
    ax[0].set_xlabel(label.format(mse_org, ssim_org))
    ax[0].set_title('Original image')
    ax[0].set_yticklabels([])
    ax[0].set_xticklabels([])

    ax[1].imshow(image_result)
    ax[1].set_xlabel(label.format(mse_org_vs_inpaint, ssim_org_vs_inpaint))
    ax[1].set_title('Original vs inpainted')
    ax[1].set_yticklabels([])
    ax[1].set_xticklabels([])

    ax[2].imshow(image_defect)
    ax[2].set_xlabel(label.format(mse_org_vs_masked, ssim_org_vs_masked))
    ax[2].set_title('Original vs noneinpaited')
    ax[2].set_yticklabels([])
    ax[2].set_xticklabels([])

    ax[3].imshow(ssim_org_full)
    ax[3].set_title('Original image')
    ax[3].set_yticklabels([])
    ax[3].set_xticklabels([])

    ax[4].imshow(ssim_org_vs_inpaint_full)
    ax[4].set_title('Original vs inpainted')
    ax[4].set_yticklabels([])
    ax[4].set_xticklabels([])

    ax[5].imshow(ssim_org_vs_masked_full)
    ax[5].set_title('Original vs noneinpaited')
    ax[5].set_yticklabels([])
    ax[5].set_xticklabels([])

    plt.tight_layout()
    plt.show()
    #pathToRidgeImages = Path.joinpath(dataPath, 'Ridgeplot')
    pathToRidgeImages = r"C:\Users\panda\PycharmProjects\Image_Inpainting_Sat\Master_Satelite_Image_Inpainting\data\generated\PartialConvolutions\12_12_2020_18_55_21\Data"
    curdatLayer = importData(config)
    images = curdatLayer.open_Imagefiles_as_array(pathToRidgeImages)
    test = SSIM_SKI.__call__(images[0],images[1])
    test2= SSIM_SKI.__call__(convertToFloat32(images[0]),convertToFloat32(images[1]))
    test3 = SSIM_SKI.__call__((images[0]*0.0255),(images[1]*0.0255))
    test4 = SSIM_SKI.__call__(images[0]/4095,images[1]/4095)
    show_images(images[0]/10000,(images[0]*0.0255),(images[0]*0.0255))
    lol =""



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main()