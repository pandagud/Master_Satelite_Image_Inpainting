import torch
import logging
import click
import os
import sys
# Set PYTHONPATH to parent folder to import module.
# (https://stackoverflow.com/questions/714063/importing-modules-from-parent-folder)
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from pathlib import Path
from src.config_default import TrainingConfig
from src.config_utillity import update_config
from polyaxon_client.tracking import get_data_paths, get_outputs_path
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
        model_path =inpainting_data_path /'models'
        modelOutputPath = Path.joinpath(model_path, 'OutputModels')
        stores_output_path = config.output_path /'data'/'storedData'
    else:
        localdir = Path().absolute().parent
        modelOutputPath = Path.joinpath(localdir, 'OutputModels')
        stores_output_path = localdir /'data'/'storedData'
    #Import test data
    test = eval_model(config)
    test.run_eval(modelOutputPath,stores_output_path)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main()