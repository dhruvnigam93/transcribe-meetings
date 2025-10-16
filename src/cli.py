"""Command-line interface for meeting transcription service."""

import click
from pathlib import Path
from loguru import logger
import sys

from .config import Config
from .pipeline import MeetingPipeline


def setup_logging(log_level: str, log_dir: Path) -> None:
    """
    Configure loguru logging.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_dir: Directory to store log files
    """
    logger.remove()  # Remove default handler

    # Console handler with colors
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level=log_level,
        colorize=True,
    )

    # File handler
    log_dir.mkdir(exist_ok=True)
    logger.add(
        log_dir / "transcribe_meetings.log",
        rotation="10 MB",
        retention="7 days",
        level="DEBUG",
    )


@click.command()
@click.option(
    "--input",
    "-i",
    "input_path",
    type=click.Path(exists=True, path_type=Path),
    help="Path to audio file or directory (defaults to 'in/' folder)",
)
@click.option(
    "--verbose", "-v", is_flag=True, help="Enable verbose logging (DEBUG level)"
)
def main(input_path: Path | None, verbose: bool) -> None:
    """
    Transcribe meeting audio and generate actionable summaries.

    Processes audio files from the 'in/' folder (or specified path) and
    generates summaries in the 'out/' folder.
    """
    # Load configuration
    config = Config.from_env()

    # Setup logging
    log_level = "DEBUG" if verbose else config.log_level
    setup_logging(log_level, config.log_dir)

    logger.info("Meeting Transcription & Summarization Service")
    logger.info("=" * 60)

    try:
        # Initialize pipeline
        pipeline = MeetingPipeline(config)

        # Determine input
        if input_path is None:
            # Process entire input directory
            logger.info(f"Processing all audio files in: {config.input_dir}")
            output_paths = pipeline.process_directory()

        elif input_path.is_file():
            # Process single file
            logger.info(f"Processing single file: {input_path}")
            output_path = pipeline.process_audio(input_path)
            output_paths = [output_path]

        elif input_path.is_dir():
            # Process specified directory
            logger.info(f"Processing all audio files in: {input_path}")
            output_paths = pipeline.process_directory(input_path)

        else:
            raise ValueError(f"Invalid input path: {input_path}")

        # Success summary
        logger.success(f"\n✓ Successfully processed {len(output_paths)} file(s)!")
        logger.success(f"✓ Summaries saved to: {config.output_dir}")

    except Exception as e:
        logger.error(f"\n✗ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
