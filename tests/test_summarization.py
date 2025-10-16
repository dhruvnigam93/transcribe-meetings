"""Test script to verify meeting summarization works correctly."""

from src.config import Config
from src.summarizer import Summarizer
from loguru import logger


def test_summarization():
    """Test the full summarization pipeline with a sample transcript."""

    # Load configuration
    config = Config.from_env()

    # Sample meeting transcript
    sample_transcript = """
    John: Good morning everyone. Let's start our weekly tech sync. 
    Sarah: Hi everyone. I wanted to discuss the new API integration we're working on.
    John: Great. What's the status?
    Sarah: We've completed the authentication layer. It uses OAuth 2.0. 
    Mike: That sounds good. Are we handling token refresh properly?
    Sarah: Yes, we implemented automatic token refresh with a 5-minute buffer.
    John: Excellent. What are the next steps?
    Sarah: We need to implement the data endpoints. I'll work on that this week.
    Mike: I can help with the error handling and retry logic.
    John: Perfect. Sarah, can you share the API documentation by Friday?
    Sarah: Yes, I'll send it out by end of day Thursday.
    John: Great. Anything else we need to discuss?
    Mike: Just a reminder that we have a code review session tomorrow at 2 PM.
    John: Thanks for the reminder. Let's wrap up then.
    """

    logger.info("Initializing summarizer...")
    summarizer = Summarizer(
        model=config.databricks_model,
        api_key=config.databricks_token,
        base_url=config.databricks_base_url,
    )

    logger.info("Generating summary for sample transcript...")
    summary = summarizer.summarize(sample_transcript)

    logger.success("✓ Summary generated successfully!")

    print("\n" + "=" * 80)
    print("SAMPLE TRANSCRIPT:")
    print("=" * 80)
    print(sample_transcript.strip())

    print("\n" + "=" * 80)
    print("GENERATED SUMMARY (STRUCTURED OUTPUT):")
    print("=" * 80)
    print(f"\nOVERALL SUMMARY:")
    print(summary.overall_summary)
    print(f"\nSUMMARY BY TOPICS:")
    print(summary.summary_by_topics)
    print(f"\nACTION ITEMS ({len(summary.action_items)} items):")
    for i, item in enumerate(summary.action_items, 1):
        print(f"  {i}. {item}")
    print("=" * 80)

    return summary


if __name__ == "__main__":
    try:
        summary = test_summarization()
        logger.success("🎉 Summarization test PASSED!")
    except Exception as e:
        logger.error(f"❌ Summarization test FAILED: {e}")
        import traceback

        traceback.print_exc()
