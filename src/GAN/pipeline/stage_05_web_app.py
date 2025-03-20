from src.GAN.config.configuration import ConfigurationManager
from src.GAN.components.web_app import GANWebApp
from src.GAN import logger

STAGE_NAME = "Web Application Deployment stage"

class WebAppDeploymentPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        # Get config that contains trained model paths
        web_app_config = config.get_web_app_config()
        web_app = GANWebApp(config=web_app_config)
        web_app.run(debug=True)

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = WebAppDeploymentPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e