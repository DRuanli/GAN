from src.GAN import logger
from src.GAN.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from src.GAN.pipeline.stage_02_prepare_gan_model import PrepareGANModelTrainingPipeline
from src.GAN.pipeline.stage_03_train_gan import GANTrainingPipeline
from src.GAN.pipeline.stage_04_evaluate_gan import GANEvaluationPipeline
from src.GAN.pipeline.stage_05_web_app import WebAppDeploymentPipeline

# Stage 5: Web Application Deployment
STAGE_NAME = "Web Application Deployment stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    web_app = WebAppDeploymentPipeline()
    web_app.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e