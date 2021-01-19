import logging
import click
import sys
import os
# Set PYTHONPATH to parent folder to import module.
# (https://stackoverflow.com/questions/714063/importing-modules-from-parent-folder)
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from src.dataLayer.importRGB import importData
from src.models.UnetPartialConvModel import generator,discriminator,criticWgan
from src.config_default import TrainingConfig
from src.config_utillity import update_config
from pathlib import Path
from polyaxon_client.tracking import get_data_paths, get_outputs_path,Experiment
from src.evalMetrics.eval_GAN_model import eval_model
from src.models.train_model import trainInpainting
from src.models.train_Wgan_model import trainInpaintingWgan




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

    curdatLayer = importData(config)
    train_countries= ['Croatia', 'Portugal', 'Serbia', 'Spain','Turkey']
    #test_countries=['Norway', 'Finland', 'Luxembourg', 'Belarus','France']
    test_countries = ['Norway']
    train, test = curdatLayer.getRGBDataLoader(trainCountries=train_countries,testCountries=test_countries)

    local_model_path = ""
    # if config.model_name == 'PartialConvolutions':
    #     curtraingModel = trainInpainting(train, test, generator, discriminator, config)
    #     local_model_path = curtraingModel.trainGAN()
    # elif config.model_name == 'PartialConvolutionsWgan':
    #     curtraingModel = trainInpaintingWgan(train, test, generator, criticWgan, config)
    #     local_model_path = curtraingModel.trainGAN()
    local_model_path = Path(r"/outputs/JacobPedersen/Satellite_Image_Inpainting_shared/groups/24/5649/models/PartialConvolutionsWgan_301.pt")
    if config.run_polyaxon:
        model_path = inpainting_data_path / 'models'
        modelOutputPath = Path.joinpath(model_path, 'OutputModels')
        stores_output_path = config.output_path / 'data' / 'storedData'
    else:
        localdir = Path().absolute().parent
        modelOutputPath = Path.joinpath(localdir, 'OutputModels')
        stores_output_path = localdir / 'data' / 'storedData'
    curevalModel = eval_model(config)
    curevalModel.run_eval(modelOutputPath, stores_output_path, model_path=local_model_path, test_dataloader=test)
    # test_countries=['Croatia', 'Portugal', 'Serbia', 'Spain','Turkey']
    # train, test = curdatLayer.getRGBDataLoader(trainCountries=train_countries,testCountries=test_countries)
    # del train
    # curevalModel.run_eval(modelOutputPath, stores_output_path, model_path=local_model_path, test_dataloader=test)



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main()