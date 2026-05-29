import os
from pathlib import Path
import pytest
import numpy as np
import rasterio
from crop_yield.data.ingestion import Sentinel1Downloader
from crop_yield.data.preprocessing import SARPreprocessor

def test_sar_preprocessor_pipeline(mock_env):
    """Test the complete preprocessing pipeline steps and intermediate results."""
    # 1. Ingest simulated raw image
    downloader = Sentinel1Downloader()
    raw_tif = downloader.download_by_aoi({}, "2023-01-01", "2023-01-02")
    assert raw_tif.exists()
    
    # 2. Preprocess
    preprocessor = SARPreprocessor(filter_window=5)
    processed_tif = preprocessor.run_pipeline(raw_tif)
    
    assert processed_tif.exists()
    assert processed_tif.name.startswith("processed_")
    
    # Check that intermediate temporary files are cleaned up
    temp_calibrated = mock_env.PROCESSED_DATA_DIR / f"temp_calibrated_{raw_tif.stem}.tif"
    temp_filtered = mock_env.PROCESSED_DATA_DIR / f"temp_filtered_{raw_tif.stem}.tif"
    temp_corrected = mock_env.PROCESSED_DATA_DIR / f"temp_corrected_{raw_tif.stem}.tif"
    
    assert not temp_calibrated.exists()
    assert not temp_filtered.exists()
    assert not temp_corrected.exists()
    
    # 3. Read processed GeoTIFF and check values
    with rasterio.open(processed_tif) as src:
        # Check projected coordinate reference system (UTM Zone 43N - EPSG:32643)
        assert src.crs.to_epsg() == 32643
        assert src.count == 2
        
        vv_processed = src.read(1)
        vh_processed = src.read(2)
        
        # Verify decibel conversion outputs negative numbers (dB backscatter)
        assert np.all(vv_processed <= 5.0)
        assert np.all(vh_processed <= 0.0)
        
        # Ensure no nan/inf values exist in the processed dataset
        assert not np.isnan(vv_processed).any()
        assert not np.isnan(vh_processed).any()
        assert not np.isinf(vv_processed).any()
        assert not np.isinf(vh_processed).any()
