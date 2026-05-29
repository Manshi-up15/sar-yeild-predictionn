# 🌾 AI Crop Yield Prediction System using SAR Satellite Imagery

A modular, production-grade agricultural forecasting platform that integrates **Sentinel-1 Synthetic Aperture Radar (SAR) imagery** and **meteorological covariates** to predict crop yield targets. 

This repository leverages traditional machine learning algorithms (Random Forest, XGBoost, Linear Regression) alongside advanced spatio-temporal Deep Learning models (CNN, CNN-LSTM) built on PyTorch. It features an automated REST API gateway built with FastAPI and a responsive, premium visualization dashboard built with Streamlit.

---

## 🚀 Key Features

* **Data Ingestion**: Multi-threaded downloader targeting the Copernicus Data Space Ecosystem (CDSE) OData APIs for raw Sentinel-1 GRD products, coupled with a meteorological ingester. Includes a robust local simulation fallback for offline development.
* **SAR Image Preprocessing**: High-fidelity raster processing pipeline using `rasterio` and `scipy`:
  - **Radiometric Calibration**: Intensity scaling to power backscatter.
  - **Speckle Noise Reduction**: Custom Lee speckle filter for noise reduction.
  - **Geometric Terrain Correction**: Re-projecting/warping WGS84 coordinates (EPSG:4326) to a projected coordinate reference system (UTM Zone 43N - EPSG:32643).
  - **Decibel (dB) Conversion & Normalization**: Backscatter coefficient scaling.
* **Feature Engineering**:
  - **Spatial statistics**: Polarization backscatter means, standard deviations, and cross-polarization $VV - VH$ ratios).
  - **Texture extraction**: Gray-Level Co-occurrence Matrix (GLCM) metrics (Contrast, Homogeneity, Energy, and Entropy) implemented natively in NumPy.
  - **Temporal dynamics**: Linear growth trends fit via least-squares regression over acquisition schedules.
  - **Weather dynamics**: Ingestion and synthesis of temperature, rainfall, and soil moisture metrics.
* **Machine Learning Engines**:
  - **Baselines**: Random Forest Regressor, XGBoost, and Linear Regression with auto-scaling safety splits.
  - **Deep Learning**: Custom CropCNN (spatial) and sequence-based CropCNNLSTM (temporal) PyTorch models with GPU/CPU acceleration.
* **REST API Gateway**: Fast, asynchronous FastAPI endpoints featuring self-healing inference (automatically triggers mock data ingestion and estimator fitting if prediction is requested before weights exist).
* **Streamlit Visualization Dashboard**: Premium dark-theme frontend rendering side-by-side backscatter maps (raw vs. preprocessed), weather analytics, training curves, and real-time yield forecasts.

---

## 📁 Repository Directory Structure

```text
sar-yield-prediction/
├── data/                      # Local data persistence (ignored in git)
│   ├── raw/                   # Ingested raw TIFFs and weather CSVs
│   ├── processed/             # Calibrated, filtered, and warped TIFFs
│   └── features/              # Tabular CSV datasets containing extracted features
├── models/                    # Saved pickle weights and PyTorch state dicts
├── src/
│   └── crop_yield/
│       ├── api/               # FastAPI backend routing and request schemas
│       │   ├── __init__.py
│       │   └── main.py
│       ├── app/               # Streamlit application frontend
│       │   ├── __init__.py
│       │   └── dashboard.py
│       ├── data/              # Ingestion and Preprocessing modules
│       │   ├── __init__.py
│       │   ├── ingestion.py
│       │   └── preprocessing.py
│       ├── evaluation/        # Validation metrics and plot scripts
│       │   ├── __init__.py
│       │   └── metrics.py
│       ├── features/          # Feature extraction pipeline (GLCM, backscatter, weather)
│       │   ├── __init__.py
│       │   └── engineering.py
│       ├── models/            # Baseline ML and Deep Learning architectures
│       │   ├── __init__.py
│       │   ├── baseline.py
│       │   ├── deep_learning.py
│       │   └── train.py
│       ├── config.py          # Pydantic Settings management
│       └── logging_utils.py   # Structured logging configuration
├── tests/                     # 34 automated unit and integration tests
├── .env.example               # Template environment configuration file
├── .gitignore                 # Restores tracking to python code directories
├── Dockerfile                 # Slim multi-port python container setup
├── docker-compose.yml         # Container orchestration configuration
├── requirements.txt           # Declared project dependencies
└── README.md                  # System Documentation (this file)
```

---

## ⚙️ Setup & Installation Instructions

Follow these step-by-step instructions to get the project running locally on your machine.

### 1. Prerequisites
Ensure you have the following installed:
* Python 3.10, 3.11, 3.12, or 3.13
* Git

---

### 2. Get the Code & Set Up Virtual Environment

Open your terminal (PowerShell/Command Prompt on Windows, or Bash on macOS/Linux) and navigate to the project directory:

```bash
# Navigate to the workspace folder
cd sar-yield-prediction

# Create a virtual environment named '.venv'
python -m venv .venv
```

#### Activate the Virtual Environment:
* **Windows (PowerShell)**:
  ```powershell
  .venv\Scripts\Activate.ps1
  ```
* **Windows (Command Prompt)**:
  ```cmd
  .venv\Scripts\activate.bat
  ```
* **macOS / Linux**:
  ```bash
  source .venv/bin/activate
  ```

---

### 3. Install Package Dependencies

Ensure your pip is up-to-date and install the required modules:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> [!NOTE]
> The dependencies include heavy scientific and geospatial packages like `rasterio`, `numpy`, `pandas`, `scikit-learn`, `xgboost`, `streamlit`, `fastapi`, and `torch`. The installation may take a few minutes depending on your internet connection.

---

### 4. Configure Environment Variables

Create a file named `.env` in the root of the project directory. You can copy the contents of `.env.example` as a template:

```env
LOG_LEVEL=INFO
COPERNICUS_USERNAME=your_copernicus_username_here
COPERNICUS_PASSWORD=your_copernicus_password_here
WEATHER_API_KEY=your_weather_api_key_here
```

> [!TIP]
> **Offline / Simulation Fallback**: If you do not have Copernicus Data Space credentials, leave the username and password fields blank or omitted. The downloader will automatically detect the absence of credentials and fall back to generating high-fidelity simulated Sentinel-1 GeoTIFF rasters locally. This allows you to explore and test the entire pipeline without needing live Copernicus accounts.

---

## 🏃 Running the Application

To run the complete platform, you need to launch both the **FastAPI Backend** and the **Streamlit Frontend**.

### 1. Launch the FastAPI Backend Server
With the virtual environment activated, start the FastAPI application using `uvicorn`:

```bash
# Add src directory to PYTHONPATH and run uvicorn
$env:PYTHONPATH="src"
python -m uvicorn crop_yield.api.main:app --host 0.0.0.0 --port 8000 --reload
```
*(On macOS/Linux, run: `PYTHONPATH=src python -m uvicorn crop_yield.api.main:app --host 0.0.0.0 --port 8000 --reload`)*

* **API Docs**: Once running, navigate to [http://localhost:8000/docs](http://localhost:8000/docs) in your browser to view the interactive Swagger/OpenAPI documentation.

---

### 2. Launch the Streamlit Dashboard
In a **new terminal window** (ensure the virtual environment is activated and you are in the project root directory), run:

```bash
# Add src directory to PYTHONPATH and run streamlit
$env:PYTHONPATH="src"
python -m streamlit run src/crop_yield/app/dashboard.py --server.port 8501
```
*(On macOS/Linux, run: `PYTHONPATH=src python -m streamlit run src/crop_yield/app/dashboard.py --server.port 8501`)*

* **Access Frontend**: Open [http://localhost:8501](http://localhost:8501) in your web browser.

---

## 🗺️ Step-by-Step Dashboard Walkthrough

Once you open the Streamlit interface, you can navigate through the pipeline using the **⚙️ Pipeline Control Panel** on the left sidebar:

1. **📊 Dashboard Overview**: View global forecasting performance metrics ($R^2$, RMSE) and inspect the historical predictions table and trends.
2. **📥 Data Ingestion**:
   - Provide a GeoJSON Area of Interest (AoI) (a default polygon for a region in Northern India is pre-filled).
   - Select start and end dates.
   - Click **Start Ingestion Pipeline**. It will fetch or simulate the Sentinel-1 VV/VH bands and weather metrics.
3. **⚙️ Run Preprocessing**:
   - View your detected raw imagery.
   - Adjust the **Lee Filter Window Size** slider (default is 5).
   - Click **Execute Preprocessing Pipeline** to run calibration, speckle filtering, EPSG:32643 projection warping, and decibel scaling.
   - View the side-by-side **Raw Intensity vs. Processed Decibel** heatmaps.
4. **🏋️ Model Training**:
   - Click **Extract Features** to process the rasters and weather data, producing an `engineered_features.csv` dataset.
   - Select a model architecture (Random Forest, XGBoost, Linear Regression, CropCNN, or CropCNNLSTM).
   - Set the training epochs and click **Fit Selected Model**. Streamlit will display the loss curves or validation metrics.
5. **🔮 Inference & Predictions**:
   - Adjust sliders for agricultural parameters (radar backscatter, GLCM textures, temperature, rainfall, soil moisture).
   - Select your trained prediction engine.
   - Click **Forecast Estimated Yield** to send a payload to the FastAPI backend and get a real-time yield forecast in tons/hectare.

---

## 🐳 Docker Container Deployment

If you prefer to run the system inside Docker, you can build and start the multi-container configuration (FastAPI + Streamlit) with a single command:

```bash
# Build and start services in the background
docker compose up --build -d
```

* **FastAPI Backend**: Exposed on [http://localhost:8000](http://localhost:8000)
* **Streamlit Dashboard**: Exposed on [http://localhost:8501](http://localhost:8501)
* **Volume Mounts**: The container mounts `./data/` and `./models/` on your host machine, ensuring all downloaded images, processed bands, tabular features, and trained weights persist locally.

To stop the containers, run:
```bash
docker compose down
```

---

## 🧪 Running the Unit Tests

The project includes **34 automated tests** validating the entire stack.
Ensure the virtual environment is activated, then run the test suite:

```bash
python -m pytest -v
```

This tests:
- Configuration and Pydantic schema validation.
- Copernicus API ingestion and fallback simulation.
- Speckel filters, calibration, and UTM warping.
- GLCM texture extraction and NumPy math correctness.
- Scikit-learn, XGBoost, and PyTorch CNN/LSTM model training loops.
- FastAPI endpoints, path resolutions, and self-healing route defaults.
- Docker configuration.

---

## 📡 REST API Quick Reference

| Method | Endpoint | Description | Payload Example |
| :--- | :--- | :--- | :--- |
| **GET** | `/` | Health and status check | *None* |
| **POST** | `/ingest` | Triggers Sentinel-1 & Weather Ingestion | `{"geojson_aoi": {...}, "start_date": "2023-01-01", "end_date": "2023-01-05"}` |
| **POST** | `/preprocess` | Calibrates, filters, and warps a raw GeoTIFF | `{"raw_path": "data/raw/sentinel1/some_scene_vv.tif"}` |
| **POST** | `/train` | Triggers feature extraction and model fitting | `{"model_type": "random_forest", "epochs": 10}` |
| **POST** | `/predict` | Predicts yield for a given set of engineered features | `{"features": {"vv_mean": -12.5, "vh_mean": -18.2, ...}, "model_type": "xgboost"}` |

> [!NOTE]
> The `/predict` endpoint is **self-healing**. If a prediction request is received for a model type that hasn't been trained yet, the API will automatically run ingestion, preprocessing, feature extraction, and model training in the background before returning the forecast, preventing 404/500 errors.
