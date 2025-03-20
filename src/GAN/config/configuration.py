import os
from src.GAN.constants import *
from src.GAN.utils.common import read_yaml, create_directories
from src.GAN.entity.config_entity import (DataIngestionConfig, PrepareGANModelConfig)

class ConfigurationManager:
    def __init__(
            self,
            config_filepath = CONFIG_FILE_PATH,
            params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir
        )

        return data_ingestion_config

    def get_prepare_gan_model_config(self) -> PrepareGANModelConfig:
        config = self.config.prepare_gan_model

        create_directories([config.root_dir])

        prepare_gan_model_config = PrepareGANModelConfig(
            root_dir=Path(config.root_dir),
            generator_model_path=Path(config.generator_model_path),
            discriminator_model_path=Path(config.discriminator_model_path),
            params_image_size=self.params.IMAGE_SIZE,
            params_learning_rate=self.params.LEARNING_RATE,
            params_generator_filters=self.params.GENERATOR_FILTERS,
            params_discriminator_filters=self.params.DISCRIMINATOR_FILTERS,
            params_use_bias=self.params.USE_BIAS,
            params_dropout_rate=self.params.DROPOUT_RATE
        )

        return prepare_gan_model_config