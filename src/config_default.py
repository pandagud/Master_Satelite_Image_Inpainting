from dataclasses import dataclass

@dataclass
class TrainingConfig:
    model_name: str = 'MISSING'
    use_gpu: bool = True
    use_multi_gpu: bool = False
    logging_verbosity: str = 'info'  # Logging verbosity (debug|info|warning|error|fatal).

    number_tiles: int = 100
    run_TCI:bool = False
    dataset_name: str = 'sataliteImage'
    loss_function: str = 'MISSING'
    epochs: int = 1000  # Number of epochs to train the model_name on.
    batch_size: int = 12  # Batch size during training.
    image_size: int = 256
    workers:int =0
    numberGPU:int = 1
    wantToLoadData:bool = True
    lr:int = 0.0002
    ngpu:int = 1
    wtl2:float = 0.999
    beta1:float = 0.5
    beta2:float = 0.999
    device:str = 'cuda'
