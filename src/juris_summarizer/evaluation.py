import json
import os

import pandas as pd
from tqdm import tqdm

from .config import RESULTS_CHECKPOINT_PATH
from .pipeline import evaluate_row


def load_completed_ids(path: str = RESULTS_CHECKPOINT_PATH) -> set:
    if not os.path.exists(path):
        return set()
    completed = set()
    with open(path) as f:
        for line in f:
            completed.add(json.loads(line)["paper_id"])
    return completed


def append_result(result: dict, path: str = RESULTS_CHECKPOINT_PATH):
    with open(path, "a") as f:
        f.write(json.dumps(result) + "\n")


def run_batch_eval(
    map_client, 
    reduce_client,
    sample_df,
    example_chunk_summaries: str,
    example_reference_summary: str,
    checkpoint_path: str = RESULTS_CHECKPOINT_PATH,
) -> pd.DataFrame:
    """
    Runs evaluate_row over sample_df, writing each result to checkpoint_path
    as it completes. Safe to re-run after a crash/rate-limit: rows already
    present in the checkpoint file are skipped, so you never pay twice for
    a row that already succeeded.
    """
    completed_ids = load_completed_ids(checkpoint_path)

    for _, row in tqdm(sample_df.iterrows(), total=len(sample_df)):
        if row["paper_id"] in completed_ids:
            continue
        try:
            result = evaluate_row(map_client, reduce_client, row, example_chunk_summaries, example_reference_summary)
            append_result(result, checkpoint_path)
        except Exception as e:
            print(f"FAILED on paper_id {row['paper_id']}: {e}")
            continue

    return pd.read_json(checkpoint_path, lines=True)
