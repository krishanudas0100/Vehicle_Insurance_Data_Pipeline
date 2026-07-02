import os
import sys

from src.cloud_storage.aws_storage import SimpleStorageService
from src.exception import MyException
from src.logger import logging
from src.entity.artifact_entity import ModelPusherArtifact, ModelEvaluationArtifact
from src.entity.config_entity import ModelPusherConfig
from src.entity.s3_estimator import Proj1Estimator
import mlflow


class ModelPusher:
    def __init__(self, model_evaluation_artifact: ModelEvaluationArtifact,
                 model_pusher_config: ModelPusherConfig):
        """
        :param model_evaluation_artifact: Output reference of data evaluation artifact stage
        :param model_pusher_config: Configuration for model pusher
        """
        self.s3 = SimpleStorageService()
        self.model_evaluation_artifact = model_evaluation_artifact
        self.model_pusher_config = model_pusher_config
        self.proj1_estimator = Proj1Estimator(bucket_name=model_pusher_config.bucket_name,
                                model_path=model_pusher_config.s3_model_key_path)

        # Reuses the same MLflow tracking URI as model_evaluation.py.
        # Set MLFLOW_TRACKING_URI / MLFLOW_TRACKING_USERNAME / MLFLOW_TRACKING_PASSWORD
        # as environment variables before running the pipeline.
        mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        if mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
            mlflow.set_experiment("vehicle-insurance-model-evaluation")

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        """
        Method Name :   initiate_model_evaluation
        Description :   This function is used to initiate all steps of the model pusher
        
        Output      :   Returns model evaluation artifact
        On Failure  :   Write an exception log and then raise an exception
        """
        logging.info("Entered initiate_model_pusher method of ModelTrainer class")

        try:
            print("------------------------------------------------------------------------------------------------")
            logging.info("Uploading artifacts folder to s3 bucket")
            
            logging.info("Uploading new model to S3 bucket....")
            self.proj1_estimator.save_model(from_file=self.model_evaluation_artifact.trained_model_path)
            model_pusher_artifact = ModelPusherArtifact(bucket_name=self.model_pusher_config.bucket_name,
                                                        s3_model_path=self.model_pusher_config.s3_model_key_path)

            logging.info("Uploaded artifacts folder to s3 bucket")
            logging.info(f"Model pusher artifact: [{model_pusher_artifact}]")

            # --- MLflow: record that this model version was deployed ---
            if os.getenv("MLFLOW_TRACKING_URI"):
                try:
                    with mlflow.start_run(run_name="model_pusher"):
                        mlflow.log_param("s3_model_path", self.model_pusher_config.s3_model_key_path)
                        mlflow.log_param("bucket_name", self.model_pusher_config.bucket_name)
                        mlflow.log_param("source_model_path", self.model_evaluation_artifact.trained_model_path)
                        mlflow.set_tag("stage", "deployed")
                        mlflow.set_tag("is_model_accepted", str(self.model_evaluation_artifact.is_model_accepted))
                        logging.info("Logged model deployment event to MLflow.")
                except Exception as mlflow_err:
                    logging.info(f"MLflow logging failed, continuing without it: {mlflow_err}")

            logging.info("Exited initiate_model_pusher method of ModelTrainer class")
            
            return model_pusher_artifact
        except Exception as e:
            raise MyException(e, sys) from e