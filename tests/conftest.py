import os
import sys
from pathlib import Path
import pytest

# Ensure the src directory is in the path
src_path = str(Path(__file__).resolve().parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

@pytest.fixture(autouse=True)
def mock_env(tmp_path):
    """
    Automatically override settings directories with a temporary directory
    to ensure tests don't modify real files.
    """
    from crop_yield.config import settings
    
    # Save original settings
    orig_base = settings.BASE_DIR
    orig_data = settings.DATA_DIR
    orig_raw = settings.RAW_DATA_DIR
    orig_processed = settings.PROCESSED_DATA_DIR
    orig_features = settings.FEATURES_DIR
    orig_models = settings.MODELS_DIR
    orig_log = settings.LOG_DIR
    
    # Set to temp path
    settings.BASE_DIR = tmp_path
    settings.DATA_DIR = tmp_path / "data"
    settings.RAW_DATA_DIR = settings.DATA_DIR / "raw"
    settings.PROCESSED_DATA_DIR = settings.DATA_DIR / "processed"
    settings.FEATURES_DIR = settings.DATA_DIR / "features"
    settings.MODELS_DIR = tmp_path / "models"
    settings.LOG_DIR = tmp_path / "logs"
    
    settings.create_directories()
    
    yield settings
    
    # Restore original settings
    settings.BASE_DIR = orig_base
    settings.DATA_DIR = orig_data
    settings.RAW_DATA_DIR = orig_raw
    settings.PROCESSED_DATA_DIR = orig_processed
    settings.FEATURES_DIR = orig_features
    settings.MODELS_DIR = orig_models
    settings.LOG_DIR = orig_log
