import pytest
from fastapi.testclient import TestClient
from crop_yield.api.main import app

@pytest.fixture
def api_client():
    return TestClient(app)

def test_api_health_check(api_client):
    """Verify health and metadata endpoints."""
    response = api_client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data


def test_api_ingestion(api_client):
    """Test full ingestion workflow trigger."""
    payload = {
        "geojson_aoi": {"type": "Polygon", "coordinates": [[[77.0, 28.0], [78.0, 28.0], [78.0, 29.0], [77.0, 29.0], [77.0, 28.0]]]},
        "start_date": "2023-05-01",
        "end_date": "2023-05-05"
    }
    response = api_client.post("/ingest", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "raw_tiff_path" in data


def test_api_preprocessing(api_client):
    """Test full preprocessing pipeline trigger."""
    response = api_client.post("/preprocess")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "processed_tiff_path" in data


def test_api_training_and_history(api_client):
    """Test model training triggers, predicting, and history checking."""
    # 1. Trigger training for baseline
    train_payload = {"model_type": "linear_regression", "epochs": 2}
    response = api_client.post("/train", json=train_payload)
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    
    # 2. Trigger training for deep learning
    dl_payload = {"model_type": "cnn", "epochs": 2}
    response = api_client.post("/train", json=dl_payload)
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    
    # 3. Test Invalid Model Type
    bad_payload = {"model_type": "invalid_model_architecture", "epochs": 2}
    response = api_client.post("/train", json=bad_payload)
    assert response.status_code == 400
    
    # 4. Fetch history
    history_resp = api_client.get("/history")
    assert history_resp.status_code == 200
    hist_data = history_resp.json()
    assert hist_data["status"] == "success"
    assert hist_data["records_count"] > 0
    
    # 5. Run prediction
    pred_payload = {
        "features": {
            "vv_mean": -12.5,
            "vv_std": 1.5,
            "vh_mean": -18.0,
            "vh_std": 1.2,
            "vv_vh_ratio": 5.5,
            "texture_contrast": 0.35,
            "texture_homogeneity": 0.6,
            "texture_energy": 0.15,
            "texture_entropy": 2.1,
            "temporal_vv_min": -13.0,
            "temporal_vv_max": -11.0,
            "temporal_vv_slope": 0.05,
            "mean_temperature": 25.0,
            "total_rainfall": 280.0,
            "mean_soil_moisture": 0.32
        }
    }
    pred_resp = api_client.post("/predict", json=pred_payload)
    assert pred_resp.status_code == 200
    assert "predicted_yield_tons_per_hectare" in pred_resp.json()
