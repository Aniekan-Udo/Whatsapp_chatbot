import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # API
    API_KEY: str
    
    # Database
    POSTGRES_URI: str
    POSTGRES_URI_POOLER: str
    
    # Redis
    CACHE_URL: str = "mem://"
    
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 4
    
    # File Upload
    UPLOAD_DIR: str = "uploads"
    MAX_FILE_SIZE_MB: int = 50
    
    # Monitoring
    PROMETHEUS_PORT: int = 9090
    OTLP_ENDPOINT: str | None = None
    
    # Environment
    ENVIRONMENT: str = "production"
    DEBUG: bool = False
    
    class Config:
        env_file = ".env"

settings = Settings()