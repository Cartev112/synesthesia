"""
Configuration management for Synesthesia BCI system.
Loads settings from environment variables.
"""

from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Application
    app_name: str = "synesthesia"
    app_version: str = "0.1.0"
    env: Literal["development", "staging", "production"] = "development"
    debug: bool = True
    
    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    
    # Database
    database_url: str = "postgresql://synesthesia:password@localhost:5432/synesthesia_db"
    database_echo: bool = False
    
    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    
    # EEG Device Settings
    eeg_device_type: Literal["simulator", "openbci", "muse"] = "simulator"
    eeg_sampling_rate: int = 256
    eeg_num_channels: int = 8
    
    # Signal Processing
    signal_window_size: float = 1.0  # seconds
    signal_overlap: float = 0.5  # 50% overlap
    bandpass_low: float = 0.5  # Hz
    bandpass_high: float = 50.0  # Hz
    notch_freq: float = 60.0  # Hz (power line noise)
    
    # Machine Learning
    ml_model_dir: str = "data/models"
    artifact_threshold: float = 0.5
    state_transition_time: float = 2.0  # seconds
    
    # Music Generation
    music_update_rate: int = 20  # Hz
    music_tempo_min: int = 80
    music_tempo_max: int = 140
    music_default_scale: str = "C_major"
    
    # Visual Parameters
    visual_update_rate: int = 20  # Hz
    
    # Data Storage
    recording_dir: str = "data/recordings"
    calibration_dir: str = "data/calibrations"
    session_retention_days: int = 30
    
    # Monitoring
    enable_metrics: bool = True
    metrics_port: int = 9090
    
    # Logging
    log_level: str = "INFO"
    log_format: Literal["json", "console"] = "json"
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.env == "development"
    
    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.env == "production"


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """
    Get settings instance.
    Useful for dependency injection in FastAPI.
    """
    return settings



