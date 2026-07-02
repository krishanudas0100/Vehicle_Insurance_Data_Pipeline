import pandas as pd
import sys
from src.configuration.mongo_db_connection import MongoDBClient

# ---- EDIT THESE ----
CSV_PATH = r"D:\MlOps-Projects\Vehicle_insurance_predection\Vehicle_Insurance_Data_Pipeline\notebook\data.csv"
DATABASE_NAME = "ml-cluster"
COLLECTION_NAME = "ml-cluste"
# ---------------------

df = pd.read_csv(CSV_PATH)
print(f"Loaded CSV with shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

mongo_client = MongoDBClient(database_name=DATABASE_NAME)
collection = mongo_client.database[COLLECTION_NAME]

records = df.to_dict(orient="records")
result = collection.insert_many(records)
print(f"Inserted {len(result.inserted_ids)} documents into '{DATABASE_NAME}.{COLLECTION_NAME}'")