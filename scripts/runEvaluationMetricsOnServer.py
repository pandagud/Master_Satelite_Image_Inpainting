import torch
import logging
import click
import os
import sys
# Set PYTHONPATH to parent folder to import module.
# (https://stackoverflow.com/questions/714063/importing-modules-from-parent-folder)
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from src.evalMetrics.FID import FIDCalculator
from src.evalMetrics.PSNR import PSNR
from src.evalMetrics.Pytorch_SSIM import ssim, SSIM
from src.evalMetrics.CC import CC
from src.evalMetrics.MAE import MSE
from src.evalMetrics.SDD import SDD
from src.evalMetrics.SSIM import SSIM_SKI
from pathlib import Path
from datetime import datetime
from src.dataLayer.importRGB import importData
from src.config_default import TrainingConfig
from src.config_utillity import update_config
from src.dataLayer import makeMasks
from tqdm.auto import tqdm
from polyaxon_client.tracking import get_data_paths, get_outputs_path
from src.shared.modelUtility import modelHelper
from src.shared.evalUtility import saveEvalToTxt
from src.models.UnetPartialConvModel import generator, Wgangenerator
from src.evalMetrics.eval_GAN_model import eval_model
@click.command()
@click.argument('args', nargs=-1)
def main(args):
    config = TrainingConfig()
    config = update_config(args, config)
    logger = logging.getLogger(__name__)

    if config.run_polyaxon:
        input_root_path = Path(get_data_paths()['data'])
        output_root_path = Path(get_outputs_path())
        inpainting_data_path = input_root_path / 'inpainting'
        os.environ['TORCH_HOME'] = str(input_root_path / 'pytorch_cache')
        config.data_path=inpainting_data_path
        config.output_path=output_root_path
        imageOutputPath = config.data_path /'data' /'generated'
        model_path =inpainting_data_path /'models'
        modelOutputPath = Path.joinpath(model_path, 'OutputModels')
        stores_output_path = config.output_path /'data'/'storedData'
    else:
        imageOutputPath = Path().absolute().parent /'data' /'generated'
        localdir = Path().absolute().parent
        modelOutputPath = Path.joinpath(localdir, 'OutputModels')
        stores_output_path = localdir /'data'/'storedData'
    #Import test data
    test = eval_model(config)
    test.run_eval(modelOutputPath,stores_output_path)
    curdatLayer = importData(config)

    train, test_dataloader = curdatLayer.getRGBDataLoader()
    del train
    test = Path.joinpath(modelOutputPath, config.model_name + '_'+str(config.epochs) + '.pt')
    print(Path.joinpath(modelOutputPath, config.model_name + '_'+str(config.epochs) + '.pt'))
    if Path.exists(Path.joinpath(modelOutputPath, config.model_name + '_'+str(config.epochs) + '.pt')):
        ##Hvis det er med wgan generator, altså layernorm, indsæt Wgangenerator istedet for generator()
        gen = generator().to(config.device)
        gen.load_state_dict(torch.load(Path.joinpath(modelOutputPath, config.model_name + '_'+str(config.epochs) + '.pt'))) ## Use epochs to identify model number
    else:
        print("Unable to find path to model")
    gen.eval()

    loadAndAgumentMasks = makeMasks.MaskClass(config,rand_seed=None,evaluation=True)
    names = []
    for i in range(len(test_dataloader.dataset.datasets)):
        # Find names of test images, in order to save the generated files with same name, for further reference
        localImg = test_dataloader.dataset.datasets[i].image_list
        # Slice string to only include the name of the file, ie after the last //
        localNames = []
        for i in localImg:
            if config.run_polyaxon:
                selected_image=i.split('/')[-1] ##Linux
            else:
                selected_image=i.split("\\")[-1]
            localNames.append(selected_image)
        names=names+localNames
    print("Found this many names "+str(len(names)))


    current_number = 0

    if not os.path.exists(Path.joinpath(imageOutputPath, config.model_name)):
        os.makedirs(Path.joinpath(imageOutputPath, config.model_name))

    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")

    local_test_path= imageOutputPath / config.model_name /  dt_string /'Data'
    local_store_path = stores_output_path / config.model_name /  dt_string /'stored_Data'
    os.makedirs(local_test_path)
    os.makedirs(local_store_path)
    start_time = datetime.now()
    testCount = 3
    for real in tqdm(test_dataloader):
        masks = loadAndAgumentMasks.returnTensorMasks(config.batch_size)
        masks = torch.from_numpy(masks)
        masks = masks.type(torch.cuda.FloatTensor)
        masks = 1 - masks
        masks.to(config.device)

        real = real.to(config.device)
        fake_masked_images = torch.mul(real, masks)
        generated_images = gen(fake_masked_images, masks)
        image_names = names[current_number:current_number+config.batch_size]
        current_number = current_number + config.batch_size ## Change naming to include all names
        #modelHelper.save_tensor_batch(generated_images,fake_masked_images,config.batch_size,path)
        for index, image in enumerate(generated_images):
            namePath= Path.joinpath(local_test_path,image_names[index])
            modelHelper.save_tensor_single(image,Path.joinpath(local_test_path, image_names[index]),raw=True)
        if testCount<0:
            break
        testCount = testCount-1
        print("Saved image to " +str(local_test_path))
    end_time = datetime.now()
    time_ran = str(end_time - start_time)
    #create dataloader with generated images
    generated_images_dataloader = curdatLayer.getGeneratedImagesDataloader(local_test_path)

    #calculate FID
    FID_Value = FIDCalculator(test_dataloader,generated_images_dataloader, len(test_dataloader)*config.batch_size, config.batch_size,config).get_FID_scores()

    #Calculate PSNR and SSIM
    dataloader_iterator = iter(generated_images_dataloader)
    psnrValues = []
    maeValues = []
    sddValues=[]
    ##ssimValues=[]
    SSIMValues = []
    CCValues = []
    #loop to calculate PSNR and SSIM for all test and generated images.
    for images_real in test_dataloader:
        try:
            images_generated  = next(dataloader_iterator)
        except StopIteration:
            dataloader_iterator  = iter(generated_images_dataloader)
            images_generated = next(dataloader_iterator)

        for index2 in range(config.batch_size):
            psnrValues.append(PSNR().__call__(images_real[index2], images_generated[index2]))
            ##CCValues.append(CC().__call__(images_real[index2], images_generated[index2]))
            maeValues.append(MSE().__call__(images_real[index2], images_generated[index2]))
            sddValues.append(SDD.__call__(images_real[index2], images_generated[index2]))
            ##ssimValues.append(SSIM.__call__(images_real[index2], images_generated[index2]))
            image1 = images_real[index2].unsqueeze(0)
            image2 = images_generated[index2].unsqueeze(0)
            SSIMValues.append(ssim(image1, image2))
        break
    meanMAE= sum(maeValues)/len(maeValues)
    minMAE = min(maeValues)
    maxMAE = max(maeValues)

    meanSDD = sum(sddValues) / len(sddValues)
    minSDD = min(sddValues)
    maxSDD = max(sddValues)

    meanPSNR = sum(psnrValues)/len(psnrValues)
    minPSNR = min(psnrValues)
    maxPSNR = max(psnrValues)

    meanSSIM = sum(SSIMValues) / len(SSIMValues)
    minSSIM = min(SSIMValues)
    maxSSIM = max(SSIMValues)

    meanCC = sum(CCValues) / len(CCValues)
    minCC = min(CCValues)
    maxCC = max(CCValues)

    #Save final results of evaluation metrics
    saveEvalToTxt(config.model_name,meanMAE,minMAE,maxMAE,meanSSIM,minSDD,maxSDD,meanSSIM,minSSIM,maxSSIM,PSNR.item(),minPSNR.item(),maxPSNR.item(),meanCC.item(),minCC.item(),maxCC.item(),FID_Value,time_ran, local_store_path)
    #Clean
    modelHelper.clearFolder(local_test_path.parent)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main()