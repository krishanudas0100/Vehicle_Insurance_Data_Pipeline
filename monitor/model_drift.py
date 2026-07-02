import sys
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from evidently import Report, DataDefinition, Dataset
from evidently.core.datasets import BinaryClassification
from evidently.presets import ClassificationPreset

from src.components.data_transformation import DataTransformation
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact

# --- Paths ---
TRAIN_PATH = r"D:\MlOps-Projects\Vehicle_insurance_predection\Vehicle_Insurance_Data_Pipeline\artifact\07_02_2026_00_54_10\data_ingestion\ingested\train.csv"
TEST_PATH  = r"D:\MlOps-Projects\Vehicle_insurance_predection\Vehicle_Insurance_Data_Pipeline\artifact\07_02_2026_00_54_10\data_ingestion\ingested\test.csv"
MODEL_PATH = r"D:\MlOps-Projects\Vehicle_insurance_predection\Vehicle_Insurance_Data_Pipeline\artifact\07_02_2026_00_54_10\model_trainer\trained_model\model.pkl"
TARGET_COL = "Response"

# --- Load data ---
reference_df = pd.read_csv(TRAIN_PATH)
current_df = pd.read_csv(TEST_PATH)

# --- Instantiate DataTransformation just to reuse its helper methods & schema config ---
# validation_report_file_path is required by the dataclass but unused by the 4 helper
# methods we're calling, so an empty string is fine here.
dt = DataTransformation(
    data_ingestion_artifact=DataIngestionArtifact(trained_file_path=TRAIN_PATH, test_file_path=TEST_PATH),
    data_transformation_config=DataTransformationConfig(),
    data_validation_artifact=DataValidationArtifact(
        validation_status=True, message="", validation_report_file_path=""
    )
)

def apply_same_transformation(df: pd.DataFrame) -> pd.DataFrame:
    """Exact same sequence as initiate_data_transformation() in data_transformation.py,
    stopping right before the ColumnTransformer step (MyModel.predict handles that part)."""
    df = dt._map_gender_column(df)
    df = dt._drop_id_column(df)
    df = dt._create_dummy_columns(df)
    df = dt._rename_columns(df)
    return df

# --- Split features/target, same as original pipeline ---
X_ref = reference_df.drop(columns=[TARGET_COL])
y_ref = reference_df[TARGET_COL]

X_cur = current_df.drop(columns=[TARGET_COL])
y_cur = current_df[TARGET_COL]

X_ref = apply_same_transformation(X_ref)
X_cur = apply_same_transformation(X_cur)

# --- Load trained model ---
# This is a MyModel instance (see estimator.py): .predict() internally runs
# preprocessing_object.transform(dataframe) then trained_model_object.predict(),
# so X_ref/X_cur must be in this feature-engineered (pre-ColumnTransformer) shape,
# which is exactly what apply_same_transformation produces. Do NOT call the
# ColumnTransformer yourself here -- MyModel already does that internally.
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

y_pred_ref = model.predict(X_ref)
y_pred_cur = model.predict(X_cur)

# --- Metrics: reference (train) vs current (test) ---
def report_metrics(y_true, y_pred, label):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n--- {label} ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    return acc, prec, rec, f1

ref_metrics = report_metrics(y_ref, y_pred_ref, "REFERENCE (train.csv)")
cur_metrics = report_metrics(y_cur, y_pred_cur, "CURRENT (test.csv)")

print("\n--- Delta (current - reference) ---")
for name, r, c in zip(["Accuracy", "Precision", "Recall", "F1"], ref_metrics, cur_metrics):
    print(f"{name}: {c - r:+.4f}")

# --- Evidently HTML report (visual dashboard) ---
# Evidently needs target + prediction columns sitting together in each dataframe
reference_df_eval = reference_df.copy()
reference_df_eval["prediction"] = y_pred_ref

current_df_eval = current_df.copy()
current_df_eval["prediction"] = y_pred_cur

data_definition = DataDefinition(
    classification=[BinaryClassification(
        target=TARGET_COL,
        prediction_labels="prediction",
    )]
)

ref_dataset = Dataset.from_pandas(reference_df_eval, data_definition=data_definition)
cur_dataset = Dataset.from_pandas(current_df_eval, data_definition=data_definition)

report = Report([ClassificationPreset()])
my_eval = report.run(current_data=cur_dataset, reference_data=ref_dataset)
my_eval.save_html(r"D:\MlOps-Projects\Vehicle_insurance_predection\Vehicle_Insurance_Data_Pipeline\monitor\model_drift_report.html")

print("\nEvidently model drift report saved at monitor\\model_drift_report.html")