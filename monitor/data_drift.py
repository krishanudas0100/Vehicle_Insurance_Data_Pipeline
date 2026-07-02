import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

# Reference = train.csv (what the model was trained on / "normal" data)
reference_df = pd.read_csv(r"D:\MlOps-Projects\Vehicle_insurance_predection\Vehicle_Insurance_Data_Pipeline\artifact\07_02_2026_00_54_10\data_ingestion\ingested\train.csv")

# Current = test.csv (what you're checking against the baseline)
current_df = pd.read_csv(r"D:\MlOps-Projects\Vehicle_insurance_predection\Vehicle_Insurance_Data_Pipeline\artifact\07_02_2026_00_54_10\data_ingestion\ingested\test.csv")

# Run the Data Drift preset
report = Report([DataDriftPreset()])
my_eval = report.run(current_data=current_df, reference_data=reference_df)

# Save as HTML report
my_eval.save_html(r"D:\MlOps-Projects\Vehicle_insurance_predection\Vehicle_Insurance_Data_Pipeline\monitor\drift_report.html")

print("Drift report saved successfully at monitor\\drift_report.html")