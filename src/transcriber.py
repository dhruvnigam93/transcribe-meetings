"""Audio transcription using OpenAI Whisper."""

from typing import Optional, Any
from pathlib import Path
import whisper
from loguru import logger


class Transcriber:
    """Handles audio transcription using OpenAI Whisper."""

    def __init__(self, model_name: str = "large-v3", device: str = "mps"):
        """
        Initialize Whisper transcriber.

        Args:
            model_name: Whisper model to use (large-v3 for best quality)
            device: Device to run model on (mps for Apple Silicon)
        """
        logger.info(f"Loading Whisper model: {model_name} on device: {device}")
        self.model = whisper.load_model(model_name, device=device)
        logger.info("Whisper model loaded successfully")

    def transcribe(
        self, audio_path: Path, language: Optional[str] = None
    ) -> dict[str, Any]:
        """
        Transcribe audio file.

        Args:
            audio_path: Path to audio file
            language: Optional language code (None for auto-detect, 'en' for English, 'hi' for Hindi)

        Returns:
            Dictionary with 'text' and metadata

        Raises:
            AssertionError: If audio file does not exist
        """
        assert audio_path.exists(), f"Audio file not found: {audio_path}"

        lang_label = language.upper() if language else "AUTO-DETECT"
        logger.info(f"Transcribing [{lang_label}]: {audio_path.name}")

        result = self.model.transcribe(
            str(audio_path),
            language=language,
            fp16=False,  # Use fp32 for better accuracy
            verbose=False,  # Reduced verbosity for cleaner logs
        )

        logger.info(
            f"Transcription complete [{lang_label}]. Detected language: {result.get('language', 'unknown')}"
        )

        return result
