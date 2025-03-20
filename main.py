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

# Stage 3: Train GAN
STAGE_NAME = "GAN Training stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    train_gan = GANTrainingPipeline()
    train_gan.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

# Stage 4: Evaluate GAN
STAGE_NAME = "GAN Evaluation stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    eval_gan = GANEvaluationPipeline()
    metrics = eval_gan.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    logger.info(f"Evaluation metrics: {metrics}")
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