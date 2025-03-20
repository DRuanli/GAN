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