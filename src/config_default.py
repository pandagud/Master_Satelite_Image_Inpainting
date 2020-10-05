from dataclasses import dataclass

@dataclass
class GeneralConfig:
    model_name: str = 'MISSING'
    use_gpu: bool = True
    use_multi_gpu: bool = False
    input_root_path: str = 'dataLayer/polyaxon/dataLayer'  # Location of input dataLayer.
    output_root_path: str = 'dataLayer/projects/prob_unet/outputs'  # Location of output dataLayer.
    logging_verbosity: str = 'info'  # Logging verbosity (debug|info|warning|error|fatal).
@dataclass
class ImportDataConfig:
    number_tiles: int = 100

@dataclass
class TrainingConfig:
    dataset_name: str = 'sataliteImage'
    loss_function: str = 'MISSING'
    epochs: int = 50  # Number of epochs to train the model_name on.
    batch_size: int = 64  # Batch size during training.
    image_size: int = 256
    workers =0
    numberGPU = 1
    wantToLoadData:bool = True
    lr:int = 0.0002
    ngpu = 1
    wtl2 = 0.999
    beta1 = 0.5
    beta2 = 0.999
    device = 'cuda'