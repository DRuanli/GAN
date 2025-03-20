from src.GAN import logger
from src.GAN.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from src.GAN.pipeline.stage_02_prepare_gan_model import PrepareGANModelTrainingPipeline

# Stage 1: Data Ingestion
STAGE_NAME = "Data Ingestion stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

# Stage 2: Prepare GAN Model
STAGE_NAME = "Prepare GAN Model stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    prepare_gan_model = PrepareGANModelTrainingPipeline()
    prepare_gan_model.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e