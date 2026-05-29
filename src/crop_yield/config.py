from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    Application settings and configurations.
    Values can be overridden using environment variables or a .env file.
    """
    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).resolve().parent.parent.parent / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Base Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
    FEATURES_DIR: Path = DATA_DIR / "features"
    MODELS_DIR: Path = BASE_DIR / "models"
    LOG_DIR: Path = BASE_DIR / "logs"

    # API Keys / Credentials
    COPERNICUS_USERNAME: Optional[str] = None
    COPERNICUS_PASSWORD: Optional[str] = None
    WEATHER_API_KEY: Optional[str] = None

    # Processing Parameters
    SPECKLE_FILTER_WINDOW: int = 5
    SPATIAL_RESOLUTION_METERS: int = 10

    # Model Parameters
    RANDOM_STATE: int = 42
    TEST_SIZE: float = 0.2
    BATCH_SIZE: int = 32
    LEARNING_RATE: float = 1e-4
    EPOCHS: int = 50

    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    # Logging Configuration
    LOG_LEVEL: str = "INFO"

    def create_directories(self) -> None:
        """Helper to create necessary directories if they don't exist."""
        for path in [
            self.DATA_DIR,
            self.RAW_DATA_DIR,
            self.PROCESSED_DATA_DIR,
            self.FEATURES_DIR,
            self.MODELS_DIR,
            self.LOG_DIR,
        ]:
            path.mkdir(parents=True, exist_ok=True)

# Instantiate settings singleton
settings = Settings()
