import os
import logging
from pathlib import Path
import pytest
from crop_yield.config import Settings
from crop_yield.logging_utils import setup_logging

def test_env_variables_override(tmp_path):
    """Verify that environment variables override configuration settings."""
    os.environ["COPERNICUS_USERNAME"] = "env_override_user"
    os.environ["COPERNICUS_PASSWORD"] = "env_override_pass"
    os.environ["SPECKLE_FILTER_WINDOW"] = "7"
    os.environ["LOG_LEVEL"] = "WARNING"
    
    # Instantiate a clean settings object to load from env
    settings_new = Settings()
    
    assert settings_new.COPERNICUS_USERNAME == "env_override_user"
    assert settings_new.COPERNICUS_PASSWORD == "env_override_pass"
    assert settings_new.SPECKLE_FILTER_WINDOW == 7
    assert settings_new.LOG_LEVEL == "WARNING"
    
    # Clean up environment variables
    del os.environ["COPERNICUS_USERNAME"]
    del os.environ["COPERNICUS_PASSWORD"]
    del os.environ["SPECKLE_FILTER_WINDOW"]
    del os.environ["LOG_LEVEL"]


def test_log_level_resolution(tmp_path):
    """Verify that string log levels from configuration map to standard logger levels."""
    # Test INFO mapping
    logger_info = setup_logging(name="test_info", log_level="INFO")
    assert logger_info.level == logging.INFO

    # Test DEBUG mapping
    logger_debug = setup_logging(name="test_debug", log_level="DEBUG")
    assert logger_debug.level == logging.DEBUG

    # Test WARNING mapping
    logger_warning = setup_logging(name="test_warning", log_level="WARNING")
    assert logger_warning.level == logging.WARNING


def test_log_file_writing(tmp_path):
    """Test that logger correctly writes outputs to the expected log directory."""
    from crop_yield.config import settings
    orig_log_dir = settings.LOG_DIR
    settings.LOG_DIR = tmp_path
    
    try:
        logger = setup_logging(name="test_write", log_level="DEBUG")
        logger.debug("Debug entry in log file.")
        
        log_file = tmp_path / "test_write.log"
        assert log_file.exists()
        
        with open(log_file, "r") as f:
            content = f.read()
        assert "Debug entry in log file." in content
    finally:
        settings.LOG_DIR = orig_log_dir
