from dataclasses import dataclass

@dataclass
class TrainingConfig:
    model_name: str = 'PartialConvolutions'
    use_gpu: bool = True
    use_multi_gpu: bool = False
    logging_verbosity: str = 'info'  # Logging verbosity (debug|info|warning|error|fatal).
    data_path:str =''
    output_path:str=""
    number_tiles: int = 1764
    run_TCI:bool = False
    dataset_name: str = 'sataliteImage'
    loss_function: str = 'MISSING'
    epochs: int = 1000  # Number of epochs to train the model_name on.
    frozenEpochs:int = 50
    batch_size: int = 12  # Batch size during training.
    image_size: int = 256
    workers:int =0
    numberGPU:int = 1
    wantToLoadData:bool = True
    lr:int = 0.0002
    ngpu:int = 1
    wtl2:float = 0.999
    beta1:float = 0.9
    beta2:float = 0.999
    device:str = 'cuda'
    trainMode:bool = True
    test_mode:bool=False
    pathToModel:str = 'm'
    n_critic = 5
    lambda_gp = 10
    save_model_step:int = 500
    save_error_step:int = 20
    run_polyaxon:bool = False
    polyaxon_experiment = None
    lambdaValid:float = 1.0
    lambdaHole:float  = 6.0
    lambdaTv:float  = 2.0
    lambdaPerceptual:float  = 0.05
    lambdaStyle:float  = 240.0
    new_generator:bool=False
    data_normalize:bool=False
    trainFrozen:bool=False
    nir_data:bool=False
    polyaxon_tracking = None