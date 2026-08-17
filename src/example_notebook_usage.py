# This is what your prompt_engineering.ipynb cells reduce to once the
# logic lives in nyaya_rag_summarizer/. Paste into notebook cells as needed —
# this file itself is just a reference, not meant to be run directly.

import pandas as pd
from juris_summarizer import (
    get_client,
    chunk_text,
    summarize_chunk,
    run_batch_eval,
)
from juris_summarizer.prompts import format_chunk_summaries

# --- Load + prep data (still notebook-level, this is EDA-adjacent) ---
df = pd.read_csv("../data/train.csv")
df["text_words"] = df["text"].apply(lambda x: len(str(x).split()))
df["summary_words"] = df["summary"].str.split().apply(len)
df["compression_ratio"] = df["summary_words"] / df["text_words"]

client = get_client()  # provider set in config.py

# --- Build the few-shot example once ---
example_idx = 713
example_reference_summary = df.loc[example_idx, "summary"]
example_chunks = chunk_text(df.loc[example_idx, "text"])
example_summaries = [
    summarize_chunk(client, c, i + 1, len(example_chunks))
    for i, c in enumerate(example_chunks)
]
example_chunk_summaries = format_chunk_summaries(example_summaries)

# --- Stratified sample across compression-ratio deciles ---
df["cr_decile"] = pd.qcut(df["compression_ratio"], 10, labels=False)
sample = df.groupby("cr_decile", group_keys=False).apply(lambda g: g.sample(1, random_state=42))

# --- Run batch eval (resumable — safe to rerun after a rate limit) ---
results_df = run_batch_eval(client, sample, example_chunk_summaries, example_reference_summary)
results_df[["generated_words", "reference_words", "rouge2_precision", "rouge2_recall", "rouge2_fmeasure"]].describe()
