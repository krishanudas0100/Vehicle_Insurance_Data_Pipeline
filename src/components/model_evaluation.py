import os
from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import ModelTrainerArtifact, DataIngestionArtifact, ModelEvaluationArtifact
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from src.exception import MyException
from src.constants import TARGET_COLUMN
from src.logger import logging
from src.utils.main_utils import load_object
import sys
import pandas as pd
from typing import Optional
from src.entity.s3_estimator import Proj1Estimator
from dataclasses import dataclass
import mlflow
import mlflow.sklearn

@dataclass
class EvaluateModelResponse:
    trained_model_f1_score: float
    best_model_f1_score: float
    is_model_accepted: bool
    difference: float


class ModelEvaluation:

    def __init__(self, model_eval_config: ModelEvaluationConfig, data_ingestion_artifact: DataIngestionArtifact,
                 model_trainer_artifact: ModelTrainerArtifact):
        try:
            self.model_eval_config = model_eval_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.model_trainer_artifact = model_trainer_artifact

            # --- MLflow / DagsHub tracking setup ---
            # Reads credentials from environment variables rather than hardcoding them.
            # Set these before running the pipeline, e.g. in PowerShell:
            #   $env:MLFLOW_TRACKING_URI = "https://dagshub.com/<user>/<repo>.mlflow"
            #   $env:MLFLOW_TRACKING_USERNAME = "<dagshub_username>"
            #   $env:MLFLOW_TRACKING_PASSWORD = "<dagshub_token>"
            mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
            if mlflow_tracking_uri:
                mlflow.set_tracking_uri(mlflow_tracking_uri)
                mlflow.set_experiment("vehicle-insurance-model-evaluation")
            else:
                logging.info("MLFLOW_TRACKING_URI not set; skipping remote MLflow tracking setup.")
        except Exception as e:
            raise MyException(e, sys) from e

    def get_best_model(self) -> Optional[Proj1Estimator]:
        """
        Method Name :   get_best_model
        Description :   This function is used to get model from production stage.
        
        Output      :   Returns model object if available in s3 storage
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            bucket_name = self.model_eval_config.bucket_name
            model_path=self.model_eval_config.s3_model_key_path
            proj1_estimator = Proj1Estimator(bucket_name=bucket_name,
                                               model_path=model_path)

            if proj1_estimator.is_model_present(model_path=model_path):
                return proj1_estimator
            return None
        except Exception as e:
            raise  MyException(e,sys)
        
    def _map_gender_column(self, df):
        """Map Gender column to 0 for Female and 1 for Male."""
        logging.info("Mapping 'Gender' column to binary values")
        df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1}).astype(int)
        return df

    def _create_dummy_columns(self, df):
        """Create dummy variables for categorical features."""
        logging.info("Creating dummy variables for categorical features")
        df = pd.get_dummies(df, drop_first=True)
        return df

    def _rename_columns(self, df):
        """Rename specific columns and ensure integer types for dummy columns."""
        logging.info("Renaming specific columns and casting to int")
        df = df.rename(columns={
            "Vehicle_Age_< 1 Year": "Vehicle_Age_lt_1_Year",
            "Vehicle_Age_> 2 Years": "Vehicle_Age_gt_2_Years"
        })
        for col in ["Vehicle_Age_lt_1_Year", "Vehicle_Age_gt_2_Years", "Vehicle_Damage_Yes"]:
            if col in df.columns:
                df[col] = df[col].astype('int')
        return df
    
    def _drop_id_column(self, df):
        """Drop the 'id' column if it exists."""
        logging.info("Dropping 'id' column")
        if "_id" in df.columns:
            df = df.drop("_id", axis=1)
        return df

    def evaluate_model(self) -> EvaluateModelResponse:
        """
        Method Name :   evaluate_model
        Description :   This function is used to evaluate trained model 
                        with production model and choose best model 
        
        Output      :   Returns bool value based on validation results
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            x, y = test_df.drop(TARGET_COLUMN, axis=1), test_df[TARGET_COLUMN]

            logging.info("Test data loaded and now transforming it for prediction...")

            x = self._map_gender_column(x)
            x = self._drop_id_column(x)
            x = self._create_dummy_columns(x)
            x = self._rename_columns(x)

            trained_model = load_object(file_path=self.model_trainer_artifact.trained_model_file_path)
            logging.info("Trained model loaded/exists.")
            trained_model_f1_score = self.model_trainer_artifact.metric_artifact.f1_score
            logging.info(f"F1_Score for this model: {trained_model_f1_score}")

            # Extra metrics on the newly trained model, purely for MLflow logging
            y_hat_trained_model = trained_model.predict(x)
            trained_model_accuracy = accuracy_score(y, y_hat_trained_model)
            trained_model_precision = precision_score(y, y_hat_trained_model)
            trained_model_recall = recall_score(y, y_hat_trained_model)

            best_model_f1_score=None
            best_model = self.get_best_model()
            if best_model is not None:
                logging.info(f"Computing F1_Score for production model..")
                y_hat_best_model = best_model.predict(x)
                best_model_f1_score = f1_score(y, y_hat_best_model)
                logging.info(f"F1_Score-Production Model: {best_model_f1_score}, F1_Score-New Trained Model: {trained_model_f1_score}")
            
            tmp_best_model_score = 0 if best_model_f1_score is None else best_model_f1_score
            result = EvaluateModelResponse(trained_model_f1_score=trained_model_f1_score,
                                           best_model_f1_score=best_model_f1_score,
                                           is_model_accepted=trained_model_f1_score > tmp_best_model_score,
                                           difference=trained_model_f1_score - tmp_best_model_score
                                           )
            logging.info(f"Result: {result}")

            # --- MLflow logging ---
            if os.getenv("MLFLOW_TRACKING_URI"):
                try:
                    with mlflow.start_run(run_name="model_evaluation"):
                        # Params: things that describe the run, not the outcome
                        mlflow.log_param("target_column", TARGET_COLUMN)
                        mlflow.log_param("test_data_path", self.data_ingestion_artifact.test_file_path)
                        mlflow.log_param("trained_model_path", self.model_trainer_artifact.trained_model_file_path)
                        mlflow.log_param("s3_model_path", self.model_eval_config.s3_model_key_path)

                        # Metrics: numeric outcomes, trackable across runs
                        mlflow.log_metric("trained_model_f1_score", trained_model_f1_score)
                        mlflow.log_metric("trained_model_accuracy", trained_model_accuracy)
                        mlflow.log_metric("trained_model_precision", trained_model_precision)
                        mlflow.log_metric("trained_model_recall", trained_model_recall)
                        if best_model_f1_score is not None:
                            mlflow.log_metric("production_model_f1_score", best_model_f1_score)
                        mlflow.log_metric("f1_score_difference", result.difference)

                        # Tags: searchable metadata
                        mlflow.set_tag("is_model_accepted", str(result.is_model_accepted))
                        mlflow.set_tag("stage", "evaluation")

                        # Log the actual model artifact for this run
                        mlflow.sklearn.log_model(trained_model, artifact_path="trained_model")

                        logging.info("Logged model evaluation run to MLflow.")
                except Exception as mlflow_err:
                    # Don't let a tracking/network failure break the pipeline itself
                    logging.info(f"MLflow logging failed, continuing without it: {mlflow_err}")

            return result

        except Exception as e:
            raise MyException(e, sys)

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        """
        Method Name :   initiate_model_evaluation
        Description :   This function is used to initiate all steps of the model evaluation
        
        Output      :   Returns model evaluation artifact
        On Failure  :   Write an exception log and then raise an exception
        """  
        try:
            print("------------------------------------------------------------------------------------------------")
            logging.info("Initialized Model Evaluation Component.")
            evaluate_model_response = self.evaluate_model()
            s3_model_path = self.model_eval_config.s3_model_key_path

            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=evaluate_model_response.is_model_accepted,
                s3_model_path=s3_model_path,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                changed_accuracy=evaluate_model_response.difference)

            logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")
            return model_evaluation_artifact
        except Exception as e:
            raise MyException(e, sys) from e