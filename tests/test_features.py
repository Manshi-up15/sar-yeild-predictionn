import os
from pathlib import Path
import pytest
import pandas as pd
from crop_yield.data.ingestion import Sentinel1Downloader, WeatherDataIngester
from crop_yield.data.preprocessing import SARPreprocessor
from crop_yield.features.engineering import FeatureExtractor

def test_feature_extractor_pipeline(mock_env):
    """Test individual feature extraction functions and the combined tabular output."""
    # 1. Setup raw ingestion and preprocessing
    raw_tif = Sentinel1Downloader().download_by_aoi({}, "2023-01-01", "2023-01-02")
    weather_csv = WeatherDataIngester().fetch_weather_for_aoi({}, "2023-01-01", "2023-01-05")
    
    preprocessed_tif = SARPreprocessor().run_pipeline(raw_tif)
    assert preprocessed_tif.exists()
    
    # 2. Extract features
    extractor = FeatureExtractor()
    
    # Test Backscatter Stats
    stats = extractor.extract_backscatter(preprocessed_tif)
    assert "vv_mean" in stats
    assert "vh_mean" in stats
    assert "vv_vh_ratio" in stats
    assert stats["vv_mean"] < 0.0
    assert stats["vh_mean"] < 0.0
    
    # Test Texture Stats
    texture = extractor.extract_texture(preprocessed_tif)
    assert "texture_contrast" in texture
    assert "texture_homogeneity" in texture
    assert "texture_energy" in texture
    assert "texture_entropy" in texture
    assert texture["texture_homogeneity"] >= 0.0
    assert texture["texture_energy"] > 0.0
    
    # Test Pipeline
    features_csv = extractor.run_pipeline(preprocessed_tif, weather_csv)
    assert features_csv.exists()
    
    # Inspect CSV
    df = pd.read_csv(features_csv)
    assert len(df) == 1
    
    # Assert columns
    cols = df.columns
    assert "vv_mean" in cols
    assert "vv_std" in cols
    assert "vh_mean" in cols
    assert "vh_std" in cols
    assert "vv_vh_ratio" in cols
    assert "texture_contrast" in cols
    assert "texture_homogeneity" in cols
    assert "texture_energy" in cols
    assert "texture_entropy" in cols
    assert "temporal_vv_min" in cols
    assert "temporal_vv_max" in cols
    assert "temporal_vv_slope" in cols
    assert "mean_temperature" in cols
    assert "total_rainfall" in cols
    assert "mean_soil_moisture" in cols
    assert "target_yield" in cols
    
    # Check yield range
    assert 0.5 <= df["target_yield"].iloc[0] <= 10.0
