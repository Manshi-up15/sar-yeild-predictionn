# 🌾 AI Crop Yield Prediction System using SAR Satellite Imagery

A modular, production-grade agricultural forecasting platform that integrates **Sentinel-1 Synthetic Aperture Radar (SAR) imagery** and **meteorological covariates** to predict crop yield targets. It uses traditional machine learning algorithms (Random Forest, XGBoost) and advanced spatio-temporal Deep Learning models (CNN, CNN-LSTM) built on PyTorch.

---

## 🚀 Key Features

* **Data Ingestion**: Multi-threaded downloader targeting Copernicus Data Space Ecosystem (CDSE) OData APIs for raw Sentinel-1 GRD products, coupled with a meteorological ingester. Includes a robust local simulation fallback for offline development.
* **SAR Image Preprocessing**: High-fidelity raster processing pipeline using `rasterio` and `scipy`:
  - Radiometric Calibration (intensity scaling).
  - Speckle Noise Reduction (custom Horizontal/Vertical Lee Filter).
  - Geometric Terrain Correction (reprojecting WGS84 coordinates to UTM Zone 43N - EPSG:32643).
  - Decibel (dB) Conversion & Normalization.
* **Feature Engineering**:
  - Spatial statistics (polarization backscatter means, standard deviations, and cross-polarization $VV - VH$ ratios).
  - Gray-Level Co-occurrence Matrix (GLCM) texture metrics (Contrast, Homogeneity, Energy, and Entropy) implemented natively in NumPy.
  - Temporal dynamics (linear growth trends fit via least-squares regression over acquisition schedules).
* **Machine Learning Engines**:
  - Baselines: Random Forest Regressor, XGBoost, and Linear Regression with auto-scaling safety splits.
  - Deep Learning: Custom CropCNN and temporal sequence CropCNNLSTM PyTorch models with GPU acceleration support.
* **REST API Gateways**: Production-grade FastAPI endpoints with self-healing inference (auto-trains default models if requested before weights exist).
* **Streamlit Visualization Dashboard**: Premium dark-theme frontend rendering side-by-side backscatter maps (raw vs. preprocessed), weather analytics, training curves, and real-time yield forecasts.

---

## 📁 Repository Directory Structure

```text
AI based/
├── data/                      # Local data persistence (ignored in git)
│   ├── raw/                   # Ingested raw TIFFs and weather CSVs
│   ├── processed/             # Calibrated and speckle-filtered TIFFs
│   └── features/              # Compiled CSV tabular dataset
├── models/                    # Saved pickle weights and PyTorch state dicts
├── src/
│   └── crop_yield/
│       ├── api/               # FastAPI routing and request schemas (main.py)
│       ├── app/               # Streamlit application logic (dashboard.py)
│       ├── data/              # Ingestion (ingestion.py) and Preprocessing (preprocessing.py)
│       ├── evaluation/        # Validation metrics and Matplotlib plot scripts (metrics.py)
│       ├── features/          # Spatial, texture, and weather extractors (engineering.py)
│       ├── models/            # Baseline (baseline.py) and PyTorch NN (deep_learning.py)
│       ├── config.py          # Pydantic Settings management
│       └── logging_utils.py   # Structured logging configuration
├── tests/                     # 34 automated unit and integration tests
├── .env.example               # Template environment configuration file
├── Dockerfile                 # Slim multi-port python container setup
├── docker-compose.yml         # Container orchestration config
├── requirements.txt           # Declared project dependencies
└── README.md                  # System Documentation (this file)
```

---

## ⚙️ Setup & Installation

### 1. Environment Configuration

Create a `.env` file in the project root folder. Copy the template from `.env.example`:

```env
LOG_LEVEL=INFO
COPERNICUS_USERNAME=your_username
COPERNICUS_PASSWORD=your_password
WEATHER_API_KEY=your_weather_key
```

*If no Copernicus credentials are provided, the downloader automatically falls back to generating geo-referenced simulated imagery, enabling offline development.*

### 2. Local Python Environment Setup

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment (Windows)
.venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 3. Running the FastAPI Backend

```bash
uvicorn src.crop_yield.api.main:app --host 0.0.0.0 --port 8000 --reload
```

*Access interactive OpenAPI docs at: http://localhost:8000/docs*

### 4. Running the Streamlit Dashboard

```bash
streamlit run src/crop_yield/app/dashboard.py --server.port 8501
```

*Access the visual interface at: http://localhost:8501*

---

## 🐳 Docker Deployment

The application is containerized to deploy the FastAPI backend and Streamlit dashboard simultaneously:

```bash
# Build and run the containers in detached mode
docker compose up --build -d
```

* FastAPI Backend is exposed on port `8000`.
* Streamlit UI is exposed on port `8501`.
* Mounted directories ensure that raw downloads, preprocessed rasters, and trained models are persisted locally on the host machine.

---

## 📡 REST API Reference

### 1. Health Check
* **Endpoint**: `GET /`
* **Response**:
  ```json
  {
    "status": "healthy",
    "service": "Crop Yield Prediction System",
    "version": "0.1.0"
  }
  ```

### 2. Trigger Ingestion
* **Endpoint**: `POST /ingest`
* **Payload**:
  ```json
  {
    "geojson_aoi": {"type": "Polygon", "coordinates": [[[77.1, 28.5], [77.3, 28.5], [77.3, 28.7], [77.1, 28.7], [77.1, 28.5]]]},
    "start_date": "2023-05-01",
    "end_date": "2023-05-05"
  }
  ```

### 3. Run Preprocessing
* **Endpoint**: `POST /preprocess`
* **Response**: Returns the workspace path of the orthorectified and Lee filtered raster.

### 4. Fit Estimators (Model Training)
* **Endpoint**: `POST /train`
* **Payload**:
  ```json
  {
    "model_type": "random_forest",
    "epochs": 10
  }
  ```
  *(Supported types: `random_forest`, `xgboost`, `linear_regression`, `cnn`, `cnn_lstm`)*

### 5. Prediction
* **Endpoint**: `POST /predict`
* **Payload**:
  ```json
  {
    "features": {
      "vv_mean": -12.5,
      "vh_mean": -18.2,
      "mean_temperature": 22.5,
      "total_rainfall": 320.0
    },
    "model_type": "random_forest"
  }
  ```
  *This endpoint is self-healing. If the model weights files do not exist, it triggers default data ingestion and estimator fitting automatically.*

---

## 🧪 Unit Testing

The workspace is protected by a suite of **34 unit and integration tests** verifying configuration settings, downloader logic, calibration loops, NumPy GLCM correctness, ML fitting, FastAPI routing, and Docker schemas.

Run the test suite using pytest:

```bash
pytest -v
```
