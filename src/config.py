"""Configuration management for the transcription service."""

from pydantic import BaseModel, Field
from pathlib import Path
import os
from dotenv import load_dotenv


class Config(BaseModel):
    """Application configuration with type safety."""

    # Paths
    input_dir: Path = Field(default=Path("in"))
    output_dir: Path = Field(default=Path("out"))
    log_dir: Path = Field(default=Path("logs"))

    # Whisper settings
    whisper_model: str = Field(default="large-v3")
    whisper_device: str = Field(default="mps")

    # Databricks/Claude settings
    databricks_token: str
    databricks_base_url: str = Field(
        default="https://dream11-e2.cloud.databricks.com/serving-endpoints"
    )
    databricks_model: str = Field(default="databricks/databricks-claude-opus-4-1")

    # Logging
    log_level: str = Field(default="INFO")

    @classmethod
    def from_env(cls) -> "Config":
        """
        Load configuration from environment variables.

        Returns:
            Config: Configuration instance populated from .env file

        Raises:
            ValidationError: If required environment variables are missing
        """
        load_dotenv()

        databricks_token = os.getenv("DATABRICKS_TOKEN")
        assert databricks_token is not None, "DATABRICKS_TOKEN must be set in .env file"

        return cls(
            databricks_token=databricks_token,
            whisper_model=os.getenv("WHISPER_MODEL", "large-v3"),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        )
