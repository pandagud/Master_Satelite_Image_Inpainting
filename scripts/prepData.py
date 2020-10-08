import logging
import click
from pathlib import Path
from src.prepData.RGBImage import prepRBGdata
from src.config_default import TrainingConfig


@click.command()
@click.argument('args', nargs=-1)
def main(args):
    """ Runs dataLayer processing scripts to turn raw dataLayer from (../raw) into
        cleaned dataLayer ready to be analyzed (saved in ../processed).
    """
    ## Talk to Rune about how dataLayer is handle. If it should be part of the "big" project.
    config = TrainingConfig()
    config = update_config(args,config)
    logger = logging.getLogger(__name__)
    logger.info('making final dataLayer set from raw dataLayer')

    curprepRBGdata= prepRBGdata(config)
    curprepRBGdata.LoadRGBdata()

def update_config(args,config):
    # Instantiate the parser
    d = dict(arg.split(':') for arg in args)
    c = config.__dict__
    for key in d.keys():
        if key in c.keys():
            newValue = d[key]
            localType = c[key]
            if isinstance(localType, int):
                c[key] = int(newValue)
            elif  isinstance(localType,bool):
                c[key] =bool(newValue)
            elif  isinstance(localType,float):
                c[key]=float(newValue)
            else:
                c[key]=newValue
    new_config = TrainingConfig(**c)
    return new_config


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main()