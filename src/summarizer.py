"""Meeting summarization using Claude via DSPy."""

import dspy
from typing import Optional
from pydantic import BaseModel, Field
from loguru import logger
import os


class MeetingSummary(BaseModel):
    """Structured meeting summary output."""

    overall_summary: str = Field(description="Overall summary of the meeting")
    key_decisions: list[str] = Field(
        description="List of key decisions made during the meeting",
        default_factory=list,
    )
    summary_by_topics: str = Field(description="Summary organized by discussion topics")
    action_items: list[str] = Field(
        description="List of action items and the people responsible for them if available",
        default_factory=list,
    )
    open_points: list[str] = Field(
        description="List of unresolved issues, pending questions, or topics requiring further discussion",
        default_factory=list,
    )


class MeetingSummarizer(dspy.Signature):
    """Summarize meeting transcript into topics, decisions, action items, and open points."""

    transcript: str = dspy.InputField(desc="Full meeting transcript")
    overall_summary: str = dspy.OutputField(
        desc="Overall summary of the meeting, including key points and context"
    )
    key_decisions: list[str] = dspy.OutputField(
        desc="List of key decisions made during the meeting with context"
    )
    summary_by_topics: str = dspy.OutputField(
        desc="Summary organized by topics/themes with bullet points"
    )
    action_items: list[str] = dspy.OutputField(
        desc="List of action items, tasks, or next steps with responsible parties if mentioned"
    )
    open_points: list[str] = dspy.OutputField(
        desc="List of unresolved issues, pending questions, or topics needing further discussion"
    )


class Summarizer:
    """Handles meeting summarization using Claude via Databricks."""

    def __init__(self, model: str, api_key: str, base_url: str):
        """
        Initialize Claude summarizer via DSPy.

        Args:
            model: Databricks model name
            api_key: Databricks API token
            base_url: Databricks serving endpoint URL
        """
        logger.info(f"Initializing DSPy with model: {model}")

        lm = dspy.LM(model, api_key=api_key, base_url=base_url)
        dspy.configure(lm=lm)

        self.predictor = dspy.ChainOfThought(MeetingSummarizer)

        logger.info("Summarizer initialized successfully")

    def summarize(self, transcript: str) -> MeetingSummary:
        """
        Generate structured summary from transcript.

        Args:
            transcript: Full meeting transcript text

        Returns:
            MeetingSummary object with overall_summary, summary_by_topics, and action_items

        Raises:
            AssertionError: If transcript is empty
        """
        assert len(transcript) > 0, "Transcript cannot be empty"

        logger.info(f"Generating summary for transcript ({len(transcript)} chars)")

        # DSPy will automatically structure the output based on the signature
        result = self.predictor(transcript=transcript)

        logger.info("Summary generated successfully")

        # Create MeetingSummary object from DSPy result
        summary = MeetingSummary(
            overall_summary=result.overall_summary,
            key_decisions=result.key_decisions if result.key_decisions else [],
            summary_by_topics=result.summary_by_topics,
            action_items=result.action_items if result.action_items else [],
            open_points=result.open_points if result.open_points else [],
        )

        return summary


if __name__ == "__main__":
    summarizer = Summarizer(
        model="databricks/databricks-claude-opus-4-1",
        api_key=os.getenv("DATABRICKS_TOKEN"),
        base_url="https://dream11-e2.cloud.databricks.com/serving-endpoints",
    )
    with open("out/20251014_Techsync_conapanion_transcript_english.txt", "r") as f:
        transcript = f.read()
    summary = summarizer.summarize(transcript=transcript)

    # Display structured output
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)
    print(summary.overall_summary)
    print("\n" + "=" * 80)
    print("KEY DECISIONS")
    print("=" * 80)
    if summary.key_decisions:
        for i, decision in enumerate(summary.key_decisions, 1):
            print(f"{i}. {decision}")
    else:
        print("No key decisions identified.")
    print("\n" + "=" * 80)
    print("SUMMARY BY TOPICS")
    print("=" * 80)
    print(summary.summary_by_topics)
    print("\n" + "=" * 80)
    print("ACTION ITEMS")
    print("=" * 80)
    if summary.action_items:
        for i, item in enumerate(summary.action_items, 1):
            print(f"{i}. {item}")
    else:
        print("No action items identified.")
    print("\n" + "=" * 80)
    print("OPEN POINTS")
    print("=" * 80)
    if summary.open_points:
        for i, point in enumerate(summary.open_points, 1):
            print(f"{i}. {point}")
    else:
        print("No open points identified.")
