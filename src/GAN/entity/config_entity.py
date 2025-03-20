# base on params.yaml and config/config.yaml do:

from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path


@dataclass(frozen=True)
class PrepareGANModelConfig:
    root_dir: Path
    generator_model_path: Path
    discriminator_model_path: Path
    params_image_size: list
    params_learning_rate: float
    params_generator_filters: int
    params_discriminator_filters: int
    params_use_bias: bool
    params_dropout_rate: float

@dataclass(frozen=True)
class GANTrainingConfig:
    root_dir: Path
    trained_generator_path: Path
    trained_discriminator_path: Path
    generator_model_path: Path
    discriminator_model_path: Path
    training_data: Path
    params_epochs: int
    params_batch_size: int
    params_image_size: list
    params_learning_rate: float
    params_beta1: float
    params_lambda_cycle: int
    params_lambda_identity: float
    params_save_interval: int

@dataclass(frozen=True)
class GANEvaluationConfig:
    generator_model_path: Path
    discriminator_model_path: Path
    test_data_path: Path
    evaluation_output_path: Path
    all_params: dict
    mlflow_uri: str
    params_image_size: list
    params_batch_size: int
    num_samples: int
    save_format: str