from src.GAN.config.configuration import ConfigurationManager
from src.GAN.components.evaluate_gan import GANEvaluation
from src.GAN import logger

STAGE_NAME = "GAN Evaluation stage"


class GANEvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        eval_config = config.get_gan_evaluation_config()
        evaluation = GANEvaluation(config=eval_config)

        metrics = evaluation.evaluate()

        try:
            evaluation.log_into_mlflow()
        except Exception as e:
            logger.warning(f"MLflow logging failed: {e}")

        return metrics


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = GANEvaluationPipeline()
        metrics = obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
        logger.info(f"Evaluation metrics: {metrics}")
    except Exception as e:
        logger.exception(e)
        raise e