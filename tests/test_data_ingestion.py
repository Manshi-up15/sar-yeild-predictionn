import os
from pathlib import Path
import pytest
import pandas as pd
import rasterio
from unittest.mock import patch, MagicMock
from crop_yield.data.ingestion import Sentinel1Downloader, WeatherDataIngester

def test_sentinel1_downloader_fallback(mock_env):
    """Test that the Sentinel-1 Downloader falls back to generating a valid GeoTIFF file."""
    downloader = Sentinel1Downloader(username=None, password=None)
    
    # Trigger fallback
    tiff_path = downloader.download_by_aoi(
        geojson_aoi={"type": "Polygon", "coordinates": [[[77.1, 28.6], [77.3, 28.6], [77.3, 28.8], [77.1, 28.8], [77.1, 28.6]]]},
        start_date="2023-01-01",
        end_date="2023-01-31"
    )
    
    assert tiff_path.exists()
    assert tiff_path.suffix == ".tif"
    
    # Load and inspect the generated GeoTIFF using rasterio
    with rasterio.open(tiff_path) as src:
        assert src.count == 2  # 2 Bands (VV and VH)
        assert src.width == 256
        assert src.height == 256
        assert src.crs.to_epsg() == 4326  # WGS 84
        
        # Read bands and verify stats
        vv_band = src.read(1)
        vh_band = src.read(2)
        assert vv_band.shape == (256, 256)
        assert vh_band.shape == (256, 256)
        
        # Check that backscatter range matches expectations
        assert vv_band.mean() < 0.0
        assert vh_band.mean() < 0.0
        
        # Check custom metadata tags
        tags = src.tags()
        assert tags.get("sensor") == "Sentinel-1A"
        assert tags.get("product_type") == "GRD"


def test_weather_ingester_records(mock_env):
    """Verify that WeatherDataIngester outputs a valid CSV dataset."""
    ingester = WeatherDataIngester()
    
    csv_path = ingester.fetch_weather_for_aoi(
        geojson_aoi={},
        start_date="2023-01-01",
        end_date="2023-01-10"
    )
    
    assert csv_path.exists()
    assert csv_path.suffix == ".csv"
    
    # Load with pandas
    df = pd.read_csv(csv_path)
    assert len(df) == 10  # 10 days of records
    assert "date" in df.columns
    assert "temperature" in df.columns
    assert "rainfall" in df.columns
    assert "soil_moisture" in df.columns
    
    # Check simple values constraints
    assert df["temperature"].mean() > 0.0
    assert df["soil_moisture"].max() <= 1.0


@patch('requests.post')
def test_sentinel1_live_auth_success(mock_post):
    """Verify Keycloak live token acquisition with CDSE credentials."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"access_token": "mock_cdse_jwt_token"}
    mock_post.return_value = mock_response
    
    downloader = Sentinel1Downloader(username="live_user", password="live_password")
    auth_result = downloader.authenticate()
    
    assert auth_result is True
    assert downloader.auth_token == "mock_cdse_jwt_token"
