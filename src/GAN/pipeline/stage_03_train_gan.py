from src.GAN.config.configuration import ConfigurationManager
from src.GAN.components.train_gan import GANTraining
from src.GAN import logger

STAGE_NAME = "GAN Training stage"

class GANTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        gan_training_config = config.get_gan_training_config()
        gan_training = GANTraining(config=gan_training_config)
        gan_training.load_models()
        gan_training.load_dataset()
        gan_training.train()

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = GANTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e