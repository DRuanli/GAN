from src.GAN import logger
from src.GAN.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from src.GAN.pipeline.stage_02_prepare_gan_model import PrepareGANModelTrainingPipeline
from src.GAN.pipeline.stage_03_train_gan import GANTrainingPipeline
from src.GAN.pipeline.stage_04_evaluate_gan import GANEvaluationPipeline
from src.GAN.pipeline.stage_05_web_app import WebAppDeploymentPipeline

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

# Stage 2: Prepare Base Model
STAGE_NAME = "Prepare GAN Model stage"
try:
    logger.info(f"*******************")
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    prepare_gan_model = PrepareGANModelTrainingPipeline()
    prepare_gan_model.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

# Stage 3: GAN Model Training
STAGE_NAME = "Training Model stage"
try:
    logger.info(f"*******************")
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    training_model = GANTrainingPipeline()
    training_model.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

# Stage 4: GAN Model Evaluation
STAGE_NAME = "Evaluation Model stage"
try:
    logger.info(f"*******************")
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    evaluate_model = GANEvaluationPipeline()
    evaluate_model.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

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