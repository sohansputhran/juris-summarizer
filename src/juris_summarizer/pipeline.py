from rouge_score import rouge_scorer

from .chunking import chunk_text
from .llm_client import summarize_chunk, reduce_summaries

_scorer = rouge_scorer.RougeScorer(["rouge2"], use_stemmer=True)


def run_pipeline(
    map_client,
    reduce_client,
    text: str,
    example_chunk_summaries: str,
    example_reference_summary: str,
) -> str:
    chunks = chunk_text(text)
    chunk_summaries = [
        summarize_chunk(map_client, c, i + 1, len(chunks)) for i, c in enumerate(chunks)
    ]
    return reduce_summaries(
        reduce_client, chunk_summaries, example_chunk_summaries, example_reference_summary
    )


def evaluate_row(
    map_client,
    reduce_client,
    row,
    example_chunk_summaries: str,
    example_reference_summary: str,
) -> dict:
    generated = run_pipeline(
        map_client, reduce_client, row["text"], example_chunk_summaries, example_reference_summary
    )
    scores = _scorer.score(row["summary"], generated)
    return {
        "paper_id": row["paper_id"],
        "text_words": row["text_words"],
        "compression_ratio": row["compression_ratio"],
        "generated": generated,
        "generated_words": len(generated.split()),
        "reference_words": row["summary_words"],
        "rouge2_precision": scores["rouge2"].precision,
        "rouge2_recall": scores["rouge2"].recall,
        "rouge2_fmeasure": scores["rouge2"].fmeasure,
    }
