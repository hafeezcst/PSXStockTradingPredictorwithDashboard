from typing import List, Union, Dict, Any
from pydantic_settings import BaseSettings
from pydantic import AnyHttpUrl, validator
import secrets
from pathlib import Path

class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str = secrets.token_urlsafe(32)
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8  # 8 days
    
    # CORS Configuration
    ALLOWED_ORIGINS: List[AnyHttpUrl] = [
        "http://localhost:3000",  # React frontend
        "http://localhost:8000",  # FastAPI backend
        "http://localhost:19006",  # React Native
    ]

    @validator("ALLOWED_ORIGINS", pre=True)
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> Union[List[str], str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)

    # Project Info
    PROJECT_NAME: str = "PSX Investment Application"
    VERSION: str = "1.0.0"
    DESCRIPTION: str = "Shariah-compliant platform for KMI-30 stock investing"

    # Database
    POSTGRES_SERVER: str = "localhost"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    POSTGRES_DB: str = "investing_script"
    SQLALCHEMY_DATABASE_URI: str = None

    @validator("SQLALCHEMY_DATABASE_URI", pre=True)
    def assemble_db_connection(cls, v: str, values: Dict[str, Any]) -> Any:
        if isinstance(v, str):
            return v
        return f"postgresql://{values.get('POSTGRES_USER')}:{values.get('POSTGRES_PASSWORD')}@{values.get('POSTGRES_SERVER')}/{values.get('POSTGRES_DB')}"

    # Redis
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: str = None

    # RabbitMQ
    RABBITMQ_HOST: str = "localhost"
    RABBITMQ_PORT: int = 5672
    RABBITMQ_USER: str = "guest"
    RABBITMQ_PASSWORD: str = "guest"

    # AI APIs
    DEEPSEEK_API_KEY: str = None
    GROK_API_KEY: str = None

    # Stripe
    STRIPE_API_KEY: str = None
    STRIPE_WEBHOOK_SECRET: str = None

    # Social Media
    TELEGRAM_BOT_TOKEN: str = None
    WHATSAPP_API_KEY: str = None
    FACEBOOK_APP_ID: str = None
    FACEBOOK_APP_SECRET: str = None
    INSTAGRAM_APP_ID: str = None
    INSTAGRAM_APP_SECRET: str = None
    TIKTOK_APP_ID: str = None
    TIKTOK_APP_SECRET: str = None

    # YouTube
    YOUTUBE_API_KEY: str = None

    # Blockchain
    ETHEREUM_NODE_URL: str = None
    ETHEREUM_PRIVATE_KEY: str = None

    # File paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    UPLOAD_DIR: Path = BASE_DIR / "uploads"
    LOG_DIR: Path = BASE_DIR / "logs"

    class Config:
        case_sensitive = True
        env_file = ".env"

settings = Settings() 