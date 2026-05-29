import os
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
import torch
from fastapi.testclient import TestClient

def test_imports():
    """Verify that all application submodules can be imported cleanly."""
    import crop_yield
    from crop_yield.config import settings
    from crop_yield.logging_utils import logger
    from crop_yield.data.ingestion import Sentinel1Downloader, WeatherDataIngester
    from crop_yield.data.preprocessing import SARPreprocessor
    from crop_yield.features.engineering import FeatureExtractor
    from crop_yield.models.baseline import BaselineModelTrainer
    from crop_yield.models.deep_learning import CropCNN, CropCNNLSTM, DeepLearningModelTrainer
    from crop_yield.evaluation.metrics import evaluate_predictions
    from crop_yield.api.main import app

    assert crop_yield.__version__ == "0.1.0"
    assert logger.name == "crop_yield"


def test_directory_creation(mock_env):
    """Ensure that the settings directories are created when requested."""
    mock_env.create_directories()
    assert mock_env.DATA_DIR.exists()
    assert mock_env.RAW_DATA_DIR.exists()
    assert mock_env.PROCESSED_DATA_DIR.exists()
    assert mock_env.FEATURES_DIR.exists()
    assert mock_env.MODELS_DIR.exists()
    assert mock_env.LOG_DIR.exists()


def test_logging(mock_env):
    """Ensure logging outputs correctly to the log file."""
    from crop_yield.logging_utils import setup_logging
    
    test_logger = setup_logging(name="test_logger")
    test_logger.info("Verifying logging system integration.")
    
    log_file = mock_env.LOG_DIR / "test_logger.log"
    assert log_file.exists()
    
    with open(log_file, "r") as f:
        content = f.read()
    assert "Verifying logging system integration" in content


def test_ingestion_and_preprocessing_skeletons(mock_env):
    """Test standard ingestion and preprocessing skeleton functions."""
    from unittest.mock import patch, MagicMock
    from crop_yield.data.ingestion import Sentinel1Downloader, WeatherDataIngester
    from crop_yield.data.preprocessing import SARPreprocessor
    
    with patch('requests.post') as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"access_token": "mock_token"}
        mock_post.return_value = mock_response
        
        downloader = Sentinel1Downloader(username="test_user", password="test_password")
        assert downloader.authenticate() is True
    
    raw_tif = downloader.download_by_aoi(
        geojson_aoi={"type": "Polygon", "coordinates": []},
        start_date="2023-01-01",
        end_date="2023-01-31"
    )
    assert raw_tif.exists()
    
    ingester = WeatherDataIngester()
    weather_csv = ingester.fetch_weather_for_aoi(
        geojson_aoi={"type": "Polygon", "coordinates": []},
        start_date="2023-01-01",
        end_date="2023-01-31"
    )
    assert weather_csv.exists()
    
    preprocessor = SARPreprocessor()
    processed_tif = preprocessor.run_pipeline(raw_tif)
    assert processed_tif.exists()


def test_feature_engineering_skeleton(mock_env):
    """Test feature engineering outputs correct pandas DataFrame columns."""
    from crop_yield.data.ingestion import Sentinel1Downloader, WeatherDataIngester
    from crop_yield.features.engineering import FeatureExtractor
    
    raw_tif = Sentinel1Downloader().download_by_aoi({}, "2023-01-01", "2023-01-31")
    weather_csv = WeatherDataIngester().fetch_weather_for_aoi({}, "2023-01-01", "2023-01-31")
    
    extractor = FeatureExtractor()
    features_file = extractor.run_pipeline(raw_tif, weather_csv)
    
    assert features_file.exists()
    df = pd.read_csv(features_file)
    assert not df.empty
    assert "vv_mean" in df.columns
    assert "vv_std" in df.columns
    assert "vh_mean" in df.columns
    assert "vh_std" in df.columns
    assert "mean_temperature" in df.columns
    assert "total_rainfall" in df.columns
    assert "target_yield" in df.columns


def test_baseline_models_skeleton():
    """Verify RandomForest and XGBoost training and prediction skeletons."""
    from crop_yield.models.baseline import BaselineModelTrainer
    
    df = pd.DataFrame({
        "vv_mean": [-12.5, -13.0, -11.8],
        "vv_std": [2.1, 1.9, 2.2],
        "vh_mean": [-18.2, -18.9, -17.5],
        "vh_std": [1.9, 1.8, 2.0],
        "mean_temperature": [22.5, 23.0, 21.8],
        "total_rainfall": [320.0, 310.0, 335.0],
        "target_yield": [3.5, 3.2, 3.8]
    })
    
    for model_type in ["random_forest", "xgboost"]:
        trainer = BaselineModelTrainer(model_type=model_type)
        summary = trainer.train(df)
        assert summary["status"] == "trained"
        
        preds = trainer.predict(df)
        assert len(preds) == 3
        
        model_path = trainer.save_model()
        assert model_path.exists()


def test_deep_learning_models():
    """Verify PyTorch CNN and CNN-LSTM architectures and forward passes."""
    from crop_yield.models.deep_learning import CropCNN, CropCNNLSTM, DeepLearningModelTrainer
    
    # 1. Test CNN
    cnn = CropCNN(in_channels=2)
    # Batch size of 4, 2 channels, 16x16 image
    dummy_x_cnn = torch.randn(4, 2, 16, 16)
    out_cnn = cnn(dummy_x_cnn)
    assert out_cnn.shape == (4, 1)
    
    # 2. Test CNN-LSTM
    cnn_lstm = CropCNNLSTM(in_channels=2)
    # Batch size of 4, sequence of 5 images, 2 channels, 16x16 image
    dummy_x_lstm = torch.randn(4, 5, 2, 16, 16)
    out_lstm = cnn_lstm(dummy_x_lstm)
    assert out_lstm.shape == (4, 1)
    
    # 3. Test Trainer
    trainer = DeepLearningModelTrainer(model_name="cnn")
    history = trainer.train_model(None, None, epochs=2)
    assert len(history["train_loss"]) == 2
    
    checkpoint_path = trainer.save_checkpoint()
    assert checkpoint_path.exists()


def test_evaluation_metrics():
    """Check regression metrics helper function."""
    from crop_yield.evaluation.metrics import evaluate_predictions
    
    y_true = np.array([3.5, 3.2, 3.8])
    y_pred = np.array([3.4, 3.3, 3.7])
    
    results = evaluate_predictions(y_true, y_pred)
    assert "rmse" in results
    assert "mae" in results
    assert "r2" in results
    assert results["rmse"] > 0
    assert results["mae"] > 0


def test_api_routes():
    """Test FastAPI application endpoints via TestClient."""
    from crop_yield.api.main import app
    
    client = TestClient(app)
    
    # Health check
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    
    # Ingestion
    ingest_payload = {
        "geojson_aoi": {"type": "Polygon", "coordinates": []},
        "start_date": "2023-01-01",
        "end_date": "2023-01-31"
    }
    response = client.post("/ingest", json=ingest_payload)
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    
    # Preprocessing
    response = client.post("/preprocess")
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    
    # Training
    train_payload = {
        "model_type": "xgboost",
        "epochs": 5
    }
    response = client.post("/train", json=train_payload)
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    
    # Prediction
    predict_payload = {
        "features": {
            "vv_mean": -12.5,
            "vh_mean": -18.2,
            "mean_temperature": 22.5,
            "total_rainfall": 320.0
        }
    }
    response = client.post("/predict", json=predict_payload)
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert "predicted_yield_tons_per_hectare" in response.json()
