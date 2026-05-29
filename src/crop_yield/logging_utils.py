import logging
import sys
from pathlib import Path
from typing import Optional, Union
from crop_yield.config import settings

def setup_logging(name: str = "crop_yield", log_level: Optional[Union[str, int]] = None) -> logging.Logger:
    """
    Configures and returns a logger instance with console and file handlers.
    """
    logger = logging.getLogger(name)
    
    # Resolve log level (default to settings.LOG_LEVEL if not specified)
    if log_level is None:
        level_str = settings.LOG_LEVEL
    else:
        level_str = log_level

    if isinstance(level_str, str):
        resolved_level = getattr(logging, level_str.upper(), logging.INFO)
    else:
        resolved_level = level_str

    logger.setLevel(resolved_level)

    # Avoid duplicate handlers if already configured
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
    )

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler (create directory if doesn't exist)
    try:
        settings.LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_file = settings.LOG_DIR / f"{name}.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        # Fallback if file directory is not writable
        print(f"Warning: Could not create log file handler: {e}", file=sys.stderr)

    return logger

# Default logger instance
logger = setup_logging()
