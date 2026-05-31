from urllib import request

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from scipy import stats
from starlette.responses import HTMLResponse, RedirectResponse
from uvicorn import run as app_run

from typing import Optional
import time

# Importing constants and pipeline modules from the project
from src.constants import APP_HOST, APP_PORT
from src.pipline.prediction_pipeline import VehicleData, VehicleDataClassifier
from src.pipline.training_pipeline import TrainPipeline

# ─────────────────────────────────────────────────────────────
# PROMETHEUS METRICS SETUP
# ─────────────────────────────────────────────────────────────
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    make_asgi_app
)

# ── 1. REQUEST METRICS ────────────────────────────────────────
# Counts every API request — only goes UP
request_counter = Counter(
    'vehicle_insurance_requests_total',
    'Total number of API requests received',
    ['method', 'endpoint', 'status_code']
    # method      → GET or POST
    # endpoint    → / or /train
    # status_code → 200, 422, 500
)

# ── 2. ERROR METRICS ──────────────────────────────────────────
# Counts every error/exception — only goes UP
error_counter = Counter(
    'vehicle_insurance_errors_total',
    'Total number of errors during prediction or training',
    ['error_type']
    # error_type → ValueError, KeyError, training_error etc.
)

# ── 3. PREDICTION RESULT METRICS ─────────────────────────────
# Counts Response-Yes vs Response-No — only goes UP
prediction_counter = Counter(
    'vehicle_insurance_predictions_total',
    'Total predictions made by result',
    ['response']
    # response → Response-Yes or Response-No
)

# ── 4. LATENCY METRICS ────────────────────────────────────────
# Tracks how long each prediction takes — histogram (buckets)
prediction_latency = Histogram(
    'vehicle_insurance_prediction_latency_seconds',
    'Time taken to process each prediction request',
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
)

# ── 5. TRAINING METRICS ───────────────────────────────────────
# Counts training runs — only goes UP
training_run_counter = Counter(
    'vehicle_insurance_training_runs_total',
    'Total number of training pipeline runs',
    ['status']
    # status → success or failure
)

# Training duration — goes UP and DOWN (Gauge)
training_duration_gauge = Gauge(
    'vehicle_insurance_training_duration_seconds',
    'Time taken for last training pipeline run'
)

# Last training timestamp — goes UP and DOWN (Gauge)
last_training_timestamp = Gauge(
    'vehicle_insurance_last_training_timestamp_seconds',
    'Unix timestamp of last successful training run'
)

# ── 6. INPUT FEATURE METRICS (for drift monitoring) ──────────
# Average value of numeric features — goes UP and DOWN (Gauge)
feature_avg_gauge = Gauge(
    'vehicle_insurance_feature_avg',
    'Rolling average of numeric input feature values',
    ['feature']
    # feature → Age, Annual_Premium, Vintage, Region_Code, Policy_Sales_Channel
)

# ── 7. DATA QUALITY METRICS ───────────────────────────────────
# Missing/null input values — only goes UP
missing_value_counter = Counter(
    'vehicle_insurance_missing_values_total',
    'Total missing or null input values detected per feature',
    ['feature']
    # feature → Gender, Age, Annual_Premium etc.
)

# Out-of-range / anomalous inputs — only goes UP
anomaly_counter = Counter(
    'vehicle_insurance_input_anomalies_total',
    'Total out-of-range or anomalous input values detected',
    ['feature']
    # feature → Age (< 18 or > 100), Annual_Premium (< 0) etc.
)

# ── 8. ACTIVE REQUESTS ────────────────────────────────────────
# Currently in-flight requests — goes UP and DOWN (Gauge)
active_requests_gauge = Gauge(
    'vehicle_insurance_active_requests',
    'Number of prediction requests currently being processed'
)

# ── 9. RESPONSE YES RATIO ─────────────────────────────────────
# Ratio of Response-Yes over total — goes UP and DOWN (Gauge)
response_yes_ratio_gauge = Gauge(
    'vehicle_insurance_response_yes_ratio',
    'Rolling ratio of Response-Yes predictions (0.0 to 1.0)'
)

# Internal counters to calculate ratio
_yes_count = 0
_total_count = 0

# ─────────────────────────────────────────────────────────────
# FASTAPI APP SETUP
# ─────────────────────────────────────────────────────────────

app = FastAPI()

# Mount Prometheus /metrics endpoint
# Access at: http://localhost:8080/metrics
metrics_app = make_asgi_app()
app.mount("/metrics/", metrics_app)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Jinja2 templates
templates = Jinja2Templates(directory='template')

# CORS
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────
# MIDDLEWARE — tracks every request automatically
# ─────────────────────────────────────────────────────────────

@app.middleware("http")
async def track_request_metrics(request: Request, call_next):
    """
    Fires on EVERY request automatically.
    Tracks: request count, latency, active requests.
    """
    # Track active requests
    active_requests_gauge.inc()

    start_time = time.time()

    response = await call_next(request)

    # Calculate latency
    latency = time.time() - start_time
    prediction_latency.observe(latency)

    # Count request with labels
    request_counter.labels(
        method=request.method,
        endpoint=request.url.path,
        status_code=str(response.status_code)
    ).inc()

    active_requests_gauge.dec()

    return response


# ─────────────────────────────────────────────────────────────
# DATA FORM CLASS
# ─────────────────────────────────────────────────────────────

class DataForm:
    """
    DataForm class to handle and process incoming form data.
    """
    def __init__(self, request: Request):
        self.request: Request = request
        self.Gender: Optional[int] = None
        self.Age: Optional[int] = None
        self.Driving_License: Optional[int] = None
        self.Region_Code: Optional[float] = None
        self.Previously_Insured: Optional[int] = None
        self.Annual_Premium: Optional[float] = None
        self.Policy_Sales_Channel: Optional[float] = None
        self.Vintage: Optional[int] = None
        self.Vehicle_Age_lt_1_Year: Optional[int] = None
        self.Vehicle_Age_gt_2_Years: Optional[int] = None
        self.Vehicle_Damage_Yes: Optional[int] = None

    async def get_vehicle_data(self):
        form = await self.request.form()
        self.Gender = form.get("Gender")
        self.Age = form.get("Age")
        self.Driving_License = form.get("Driving_License")
        self.Region_Code = form.get("Region_Code")
        self.Previously_Insured = form.get("Previously_Insured")
        self.Annual_Premium = form.get("Annual_Premium")
        self.Policy_Sales_Channel = form.get("Policy_Sales_Channel")
        self.Vintage = form.get("Vintage")
        self.Vehicle_Age_lt_1_Year = form.get("Vehicle_Age_lt_1_Year")
        self.Vehicle_Age_gt_2_Years = form.get("Vehicle_Age_gt_2_Years")
        self.Vehicle_Damage_Yes = form.get("Vehicle_Damage_Yes")


# ─────────────────────────────────────────────────────────────
# HELPER: Validate & Track Input Features
# ─────────────────────────────────────────────────────────────

def track_input_metrics(form: DataForm):
    """
    Checks for:
    1. Missing/null values       → missing_value_counter
    2. Out-of-range values       → anomaly_counter
    3. Feature averages          → feature_avg_gauge (for drift)
    """
    global _yes_count, _total_count

    # ── Missing value checks ──────────────────────────────────
    all_fields = {
        "Gender": form.Gender,
        "Age": form.Age,
        "Driving_License": form.Driving_License,
        "Region_Code": form.Region_Code,
        "Previously_Insured": form.Previously_Insured,
        "Annual_Premium": form.Annual_Premium,
        "Policy_Sales_Channel": form.Policy_Sales_Channel,
        "Vintage": form.Vintage,
        "Vehicle_Age_lt_1_Year": form.Vehicle_Age_lt_1_Year,
        "Vehicle_Age_gt_2_Years": form.Vehicle_Age_gt_2_Years,
        "Vehicle_Damage_Yes": form.Vehicle_Damage_Yes,
    }

    for field, value in all_fields.items():
        if value is None or value == "":
            # Missing value detected → increment counter
            missing_value_counter.labels(feature=field).inc()

    # ── Anomaly / out-of-range checks ────────────────────────
    try:
        if form.Age and (int(form.Age) < 18 or int(form.Age) > 100):
            anomaly_counter.labels(feature='Age').inc()
    except (ValueError, TypeError):
        anomaly_counter.labels(feature='Age').inc()

    try:
        if form.Annual_Premium and float(form.Annual_Premium) < 0:
            anomaly_counter.labels(feature='Annual_Premium').inc()
    except (ValueError, TypeError):
        anomaly_counter.labels(feature='Annual_Premium').inc()

    try:
        if form.Vintage and (int(form.Vintage) < 0 or int(form.Vintage) > 365):
            anomaly_counter.labels(feature='Vintage').inc()
    except (ValueError, TypeError):
        anomaly_counter.labels(feature='Vintage').inc()

    try:
        if form.Region_Code and (float(form.Region_Code) < 0 or float(form.Region_Code) > 60):
            anomaly_counter.labels(feature='Region_Code').inc()
    except (ValueError, TypeError):
        anomaly_counter.labels(feature='Region_Code').inc()

    # ── Feature average tracking (for drift detection) ───────
    numeric_features = {
        'Age': form.Age,
        'Annual_Premium': form.Annual_Premium,
        'Vintage': form.Vintage,
        'Region_Code': form.Region_Code,
        'Policy_Sales_Channel': form.Policy_Sales_Channel,
    }

    for feature, value in numeric_features.items():
        try:
            if value is not None and value != "":
                feature_avg_gauge.labels(feature=feature).set(float(value))
        except (ValueError, TypeError):
            pass


# ─────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────

# GET / — render the form
@app.get("/", tags=["authentication"])
async def vehicledata(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="vehicledata.html"
    )


# POST / — prediction route with full metrics
@app.post("/")
async def predictRouteClient(request: Request):
    """
    Receives form data → validates → predicts → tracks all metrics.
    """
    global _yes_count, _total_count

    try:
        # ── Load form data ────────────────────────────────────
        form = DataForm(request)
        await form.get_vehicle_data()

        # ── Track input quality + feature averages ────────────
        track_input_metrics(form)

        # ── Build VehicleData object ──────────────────────────
        vehicle_data = VehicleData(
            Gender=form.Gender,
            Age=form.Age,
            Driving_License=form.Driving_License,
            Region_Code=form.Region_Code,
            Previously_Insured=form.Previously_Insured,
            Annual_Premium=form.Annual_Premium,
            Policy_Sales_Channel=form.Policy_Sales_Channel,
            Vintage=form.Vintage,
            Vehicle_Age_lt_1_Year=form.Vehicle_Age_lt_1_Year,
            Vehicle_Age_gt_2_Years=form.Vehicle_Age_gt_2_Years,
            Vehicle_Damage_Yes=form.Vehicle_Damage_Yes
        )

        # ── Run prediction ────────────────────────────────────
        vehicle_df = vehicle_data.get_vehicle_input_data_frame()
        model_predictor = VehicleDataClassifier()
        value = model_predictor.predict(dataframe=vehicle_df)[0]
        status = "Response-Yes" if value == 1 else "Response-No"

        # ── Track prediction result ───────────────────────────
        prediction_counter.labels(response=status).inc()

        # ── Update Response-Yes ratio ─────────────────────────
        _total_count += 1
        if value == 1:
            _yes_count += 1
        response_yes_ratio_gauge.set(_yes_count / _total_count)

        return templates.TemplateResponse(
            request=request,
            name="vehicledata.html",
            context={"context": status}
        )

    except Exception as e:
        import traceback
        # ── Track error with type ─────────────────────────────
        error_counter.labels(error_type=type(e).__name__).inc()
        return {"status": False, "error": traceback.format_exc()}


# GET /train — training route with full metrics
@app.get("/train")
async def trainRouteClient():
    """
    Triggers training pipeline and tracks duration, status, timestamp.
    """
    start_time = time.time()
    try:
        train_pipeline = TrainPipeline()
        train_pipeline.run_pipeline()

        # ── Track training success ────────────────────────────
        duration = time.time() - start_time
        training_duration_gauge.set(duration)
        training_run_counter.labels(status='success').inc()
        last_training_timestamp.set(time.time())

        return Response("Training successful!!!")

    except Exception as e:
        # ── Track training failure ────────────────────────────
        training_run_counter.labels(status='failure').inc()
        error_counter.labels(error_type='training_error').inc()
        return Response(f"Error Occurred! {e}")


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)