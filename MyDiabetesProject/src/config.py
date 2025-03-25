import os
import pathlib
from pydantic_settings import BaseSettings

class CommonSettings(BaseSettings):
    PROJECT_NAME: str = "MyDiabetesProject"
    BASE_DIR: str = str(pathlib.Path(__file__).parent.resolve())
    DATA_PATH: str = str(pathlib.Path(BASE_DIR, "../data/processed/diabetes.csv").resolve())
    MODEL_PATH: str = str(pathlib.Path(BASE_DIR, "../models/diabetes_model.pkl").resolve())
    TEST_RATIO: float = 0.2
    RANDOM_STATE: int = 42
    BATCH_SIZE: int = 32
    EPOCHS: int = 10
    LEARNING_RATE: float = 0.001
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_NAME: str = "diabetesdb"
    DB_USER: str = "dbuser"
    DB_PASSWORD: str = "dbpass"
    LOG_LEVEL: str = "INFO"
    MODEL_VERSION: str = "1.0.0"
    SERVICE_URL: str = "http://localhost:5000"
    SENTRY_DSN: str = ""
    REDIS_URL: str = ""
    CACHE_TTL: int = 300
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

class DevSettings(CommonSettings):
    ENV_NAME: str = "development"
    DEBUG: bool = True

class TestSettings(CommonSettings):
    ENV_NAME: str = "test"
    DEBUG: bool = True

class StagingSettings(CommonSettings):
    ENV_NAME: str = "staging"
    DEBUG: bool = False

class ProdSettings(CommonSettings):
    ENV_NAME: str = "production"
    DEBUG: bool = False

def get_settings():
    e = os.getenv("ENV_NAME", "development").strip().lower()
    if e == "production":
        return ProdSettings()
    if e == "staging":
        return StagingSettings()
    if e == "test":
        return TestSettings()
    if e == "development":
        return DevSettings()
    return DevSettings()

settings = get_settings()
