import os
from src.GAN.constants import *
from src.GAN.utils.common import read_yaml, create_directories
from src.GAN.entity.config_entity import (DataIngestionConfig, PrepareGANModelConfig, GANTrainingConfig)


class ConfigurationManager:
    def __init__(
            self,
            config_filepath=CONFIG_FILE_PATH,
            params_filepath=PARAMS_FILE_PATH):
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
        params = self.params

        create_directories([config.root_dir])

        prepare_gan_model_config = PrepareGANModelConfig(
            root_dir=config.root_dir,
            generator_model_path=config.generator_model_path,
            discriminator_model_path=config.discriminator_model_path,
            params_image_size=params.IMAGE_SIZE,
            params_learning_rate=params.LEARNING_RATE,
            params_generator_filters=params.GENERATOR_FILTERS,
            params_discriminator_filters=params.DISCRIMINATOR_FILTERS,
            params_use_bias=params.USE_BIAS,
            params_dropout_rate=params.DROPOUT_RATE
        )

        return prepare_gan_model_config

    def get_gan_training_config(self) -> GANTrainingConfig:
        training = self.config.gan_training
        prepare_gan_model = self.config.prepare_gan_model
        params = self.params

        # Look for data in standard data ingestion paths
        base_data_path = self.config.data_ingestion.unzip_dir

        create_directories([Path(training.root_dir)])

        gan_training_config = GANTrainingConfig(
            root_dir=Path(training.root_dir),
            trained_generator_path=Path(training.trained_generator_path),
            trained_discriminator_path=Path(training.trained_discriminator_path),
            generator_model_path=Path(prepare_gan_model.generator_model_path),
            discriminator_model_path=Path(prepare_gan_model.discriminator_model_path),
            training_data=Path(base_data_path),
            params_epochs=params.EPOCHS,
            params_batch_size=params.BATCH_SIZE,
            params_image_size=params.IMAGE_SIZE,
            params_learning_rate=params.LEARNING_RATE,
            params_beta1=params.BETA1,
            params_lambda_cycle=params.LAMBDA_CYCLE,
            params_lambda_identity=params.LAMBDA_IDENTITY,
            params_save_interval=params.SAVE_INTERVAL
        )

        return gan_training_config