"""Pipeline orchestration for transcription and summarization."""

from pathlib import Path
from typing import Optional
from loguru import logger

from .transcriber import Transcriber
from .summarizer import Summarizer
from .config import Config


class MeetingPipeline:
    """Orchestrates the meeting transcription and summarization pipeline."""

    def __init__(self, config: Config):
        """
        Initialize pipeline with configuration.

        Args:
            config: Application configuration
        """
        self.config = config
        self.transcriber: Optional[Transcriber] = None
        self.summarizer: Optional[Summarizer] = None

        # Ensure output directories exist
        self.config.output_dir.mkdir(exist_ok=True)
        self.config.log_dir.mkdir(exist_ok=True)

    def _init_transcriber(self) -> None:
        """Lazy initialization of transcriber."""
        if self.transcriber is None:
            self.transcriber = Transcriber(
                model_name=self.config.whisper_model, device=self.config.whisper_device
            )

    def _init_summarizer(self) -> None:
        """Lazy initialization of summarizer."""
        if self.summarizer is None:
            self.summarizer = Summarizer(
                model=self.config.databricks_model,
                api_key=self.config.databricks_token,
                base_url=self.config.databricks_base_url,
            )

    def _format_summary(self, summary) -> str:
        """
        Format a MeetingSummary object into readable text.

        Args:
            summary: MeetingSummary object with structured fields

        Returns:
            Formatted summary text
        """
        formatted = []

        # Overall Summary
        formatted.append("=" * 80)
        formatted.append("OVERALL SUMMARY")
        formatted.append("=" * 80)
        formatted.append(summary.overall_summary)
        formatted.append("")

        # Key Decisions
        formatted.append("=" * 80)
        formatted.append("KEY DECISIONS")
        formatted.append("=" * 80)
        if summary.key_decisions:
            for i, decision in enumerate(summary.key_decisions, 1):
                formatted.append(f"{i}. {decision}")
        else:
            formatted.append("No key decisions identified.")
        formatted.append("")

        # Summary by Topics
        formatted.append("=" * 80)
        formatted.append("SUMMARY BY TOPICS")
        formatted.append("=" * 80)
        formatted.append(summary.summary_by_topics)
        formatted.append("")

        # Action Items
        formatted.append("=" * 80)
        formatted.append("ACTION ITEMS")
        formatted.append("=" * 80)
        if summary.action_items:
            for i, item in enumerate(summary.action_items, 1):
                formatted.append(f"{i}. {item}")
        else:
            formatted.append("No action items identified.")
        formatted.append("")

        # Open Points
        formatted.append("=" * 80)
        formatted.append("OPEN POINTS")
        formatted.append("=" * 80)
        if summary.open_points:
            for i, point in enumerate(summary.open_points, 1):
                formatted.append(f"{i}. {point}")
        else:
            formatted.append("No open points identified.")
        formatted.append("")

        return "\n".join(formatted)

    def process_audio(self, audio_path: Path) -> Path:
        """
        Process a single audio file through the complete pipeline.
        Runs Hindi and English transcriptions in parallel for better accuracy.

        Args:
            audio_path: Path to input audio file

        Returns:
            Path to generated summary file

        Raises:
            AssertionError: If audio file does not exist
        """
        assert audio_path.exists(), f"Audio file not found: {audio_path}"

        logger.info(f"Starting pipeline for: {audio_path.name}")

        # Step 1: Parallel Transcription (Hindi + English)
        logger.info("=" * 50)
        logger.info("STEP 1: DUAL TRANSCRIPTION (HINDI + ENGLISH)")
        logger.info("=" * 50)

        self._init_transcriber()

        # Run both transcriptions (sequentially for model thread-safety)
        transcripts = {}

        logger.info("Running dual-language transcriptions...")

        # Hindi transcription
        logger.info("Starting Hindi transcription...")
        hindi_result = self.transcriber.transcribe(audio_path, language="hi")
        transcripts["hi"] = hindi_result["text"]
        logger.success("✓ Hindi transcription completed")

        # English transcription
        logger.info("Starting English transcription...")
        english_result = self.transcriber.transcribe(audio_path, language="en")
        transcripts["en"] = english_result["text"]
        logger.success("✓ English transcription completed")

        # Save individual transcripts for debugging
        hindi_transcript_path = (
            self.config.output_dir / f"{audio_path.stem}_transcript_hindi.txt"
        )
        english_transcript_path = (
            self.config.output_dir / f"{audio_path.stem}_transcript_english.txt"
        )

        hindi_transcript_path.write_text(transcripts["hi"], encoding="utf-8")
        english_transcript_path.write_text(transcripts["en"], encoding="utf-8")

        logger.info(f"Hindi transcript saved to: {hindi_transcript_path}")
        logger.info(f"English transcript saved to: {english_transcript_path}")

        # Step 2: Summarization with both transcripts
        logger.info("=" * 50)
        logger.info("STEP 2: SUMMARIZATION (USING BOTH TRANSCRIPTS)")
        logger.info("=" * 50)

        self._init_summarizer()

        # Create combined prompt with both transcripts
        combined_input = f"""I have two transcriptions of the same meeting audio - one in Hindi and one in English.
Please analyze BOTH transcriptions to create the most accurate and comprehensive summary.

HINDI TRANSCRIPTION:
{transcripts["hi"]}

ENGLISH TRANSCRIPTION:
{transcripts["en"]}

Please provide a unified summary that captures all important information from both transcriptions."""

        summary_obj = self.summarizer.summarize(combined_input)

        # Format the structured summary into readable text
        formatted_summary = self._format_summary(summary_obj)

        # Save final summary
        output_path = self.config.output_dir / f"{audio_path.stem}.txt"
        output_path.write_text(formatted_summary, encoding="utf-8")

        logger.info("=" * 50)
        logger.info(f"✓ Pipeline complete! Summary saved to: {output_path}")
        logger.info("=" * 50)

        return output_path

    def process_directory(self, input_dir: Optional[Path] = None) -> list[Path]:
        """
        Process all audio files in a directory.

        Args:
            input_dir: Directory to process (defaults to config.input_dir)

        Returns:
            List of generated summary file paths

        Raises:
            AssertionError: If no audio files found in directory
        """
        input_dir = input_dir or self.config.input_dir

        # Common audio extensions supported by Whisper
        audio_extensions = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus", ".webm"}

        audio_files = [
            f
            for f in input_dir.iterdir()
            if f.is_file() and f.suffix.lower() in audio_extensions
        ]

        assert len(audio_files) > 0, f"No audio files found in {input_dir}"

        logger.info(f"Found {len(audio_files)} audio file(s) to process")

        output_paths = []
        for audio_file in audio_files:
            try:
                output_path = self.process_audio(audio_file)
                output_paths.append(output_path)
            except Exception as e:
                logger.error(f"Failed to process {audio_file.name}: {e}")
                raise

        return output_paths
