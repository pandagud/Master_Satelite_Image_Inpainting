import logging
import click
import sys
import os
import torch
# Set PYTHONPATH to parent folder to import module.
# (https://stackoverflow.com/questions/714063/importing-modules-from-parent-folder)
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from src.dataLayer.importRGB import importData
from polyaxon_client.tracking import get_data_paths, get_outputs_path,Experiment
from pathlib import Path
from datetime import datetime
from src.dataLayer.importRGB import importData
from src.config_default import TrainingConfig
from src.config_utillity import update_config
from src.dataLayer import makeMasks
from tqdm.auto import tqdm
from polyaxon_client.tracking import get_data_paths, get_outputs_path
from src.shared.modelUtility import modelHelper
from src.models.UnetPartialConvModel import generator, Wgangenerator
from src.models.UnetPartialConvModelNIR import generatorNIR,Wgangenerator



@click.command()
@click.argument('args', nargs=-1)
def main(args):
    """ Runs dataLayer processing scripts to turn raw dataLayer from (../raw) into
        cleaned dataLayer ready to be analyzed (saved in ../processed).
    """
    ## Talk to Rune about how dataLayer is handle.
    config = TrainingConfig()
    config = update_config(args,config)
    ## For polyaxon
    if config.run_polyaxon:
        input_root_path = Path(get_data_paths()['data'])
        output_root_path = Path(get_outputs_path())
        inpainting_data_path = input_root_path / 'inpainting'
        os.environ['TORCH_HOME'] = str(input_root_path / 'pytorch_cache')
        config.data_path=inpainting_data_path
        config.output_path=output_root_path
        config.polyaxon_experiment=Experiment()

    logger = logging.getLogger(__name__)
    logger.info('making final dataLayer set from raw dataLayer')

    curdatLayer = importData(config)
    if config.nir_data:
        train, test_dataloader = curdatLayer.getNIRDataLoader()
    else:
        train, test_dataloader = curdatLayer.getRGBDataLoader()
    local_model_path= r"C:\Users\panda\PycharmProjects\Image_Inpainting_Sat\Master_Satelite_Image_Inpainting\OutputModels\PartialConvolutionsWgan_301.pt"
    local_output_path =Path(r"E:\Speciale\final_model")
    #gen = Wgangenerator().to(config.device)
    if config.nir_data:
        gen = generatorNIR().to(config.device)
    else:
        gen = generator().to(config.device)
    gen.load_state_dict(torch.load(local_model_path))  ## Use epochs to identify model number
    gen.eval()

    loadAndAgumentMasks = makeMasks.MaskClass(config, rand_seed=None, evaluation=True,noFlip=True)
    names = []
    # Find names of test images, in order to save the generated files with same name, for further reference
    localImg = test_dataloader.dataset.image_list
    # Slice string to only include the name of the file, ie after the last //
    localNames = []
    # if self.config.run_polyaxon:
    #     split_path = localImg[0].split('/')  ##Linux
    # else:
    #     split_path = localImg[0].split("\\")
    # local_index= split_path.index('processed')
    # local_country= split_path[local_index+1]
    for i in localImg:
        if config.run_polyaxon:
            selected_image = i.split('/')[-1]  ##Linux
        else:
            selected_image = i.split("\\")[-1]
        localNames.append(selected_image)
    names = names + localNames

    print("Found this many names " + str(len(names)))

    current_number = 0

    if not os.path.exists(Path.joinpath(local_output_path, config.model_name)):
        os.makedirs(Path.joinpath(local_output_path, config.model_name))

    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")

    local_test_path = local_output_path / config.model_name / dt_string / 'Data'
    local_test_nir_path = local_output_path / config.model_name / dt_string / 'DataNir'
    local_store_path = local_output_path / config.model_name / dt_string / 'stored_Data'
    os.makedirs(local_test_path)
    if config.nir_data:
        os.makedirs(local_test_nir_path)
    start_time = datetime.now()
    for real in tqdm(test_dataloader, disable=config.run_polyaxon):
        masks = loadAndAgumentMasks.returnTensorMasks(config.batch_size)
        masks = torch.from_numpy(masks)
        masks = masks.type(torch.cuda.FloatTensor)
        masks = 1 - masks
        masks.to(config.device)

        real = real.to(config.device)
        fake_masked_images = torch.mul(real, masks)
        generated_images = gen(fake_masked_images, masks)
        image_names = names[current_number:current_number + config.batch_size]
        current_number = current_number + config.batch_size  ## Change naming to include all names
        # modelHelper.save_tensor_batch(generated_images,fake_masked_images,config.batch_size,path)
        if config.nir_data:
            for index, image in enumerate(generated_images):
                namePath = Path.joinpath(local_test_path, image_names[index])
                if config.nir_data:
                    modelHelper.save_tensor_single_NIR(image, Path.joinpath(local_test_path, image_names[index]),
                                                       Path.joinpath(local_test_nir_path, image_names[index]), raw=True)
        else:
            modelHelper.save_tensor_batch(real, fake_masked_images, generated_images, config.batch_size,
                                      Path.joinpath(local_test_path, "_final_model_"+str(current_number)))

        current_number = current_number+1
    end_time = datetime.now()


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main()