from src.GAN.config.configuration import ConfigurationManager
from src.GAN.components.prepare_gan_model import PrepareGANModel
from src.GAN import logger

STAGE_NAME = "Prepare GAN Model Stage"

class PrepareGANModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        prepare_gan_model_config = config.get_prepare_gan_model_config()
        prepare_gan_model = PrepareGANModel(config=prepare_gan_model_config)
        generator, discriminator = prepare_gan_model.prepare_gan_model()

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = PrepareGANModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e