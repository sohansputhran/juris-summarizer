"""
Central config. Change the model/provider here rather than editing
individual functions — providers and model availability have been
churning (Groq deprecations, Cerebras catalog narrowing), so keep this
as the single point of change.
"""

# --- Provider / model ---
# "groq" or "cerebras". Both currently point at gpt-oss models.
MAP_PROVIDER = "groq"
MAP_MODEL = "openai/gpt-oss-20b"

REDUCE_PROVIDER = "cerebras"
REDUCE_MODEL = "gpt-oss-120b"

# --- Chunking ---
CHUNK_MAX_TOKENS = 2500

# --- Generation ---
# gpt-oss models spend part of max_tokens on internal reasoning tokens
# before writing the visible answer — keep this generous or you'll get
# empty/truncated completions (finish_reason == "length").
MAP_MAX_TOKENS = 1500
REDUCE_MAX_TOKENS = 1500
TEMPERATURE = 0.3
REASONING_EFFORT = "low"  # keep low: task doesn't need heavy reasoning, and it's billed

# --- Tokenizer ---
TIKTOKEN_ENCODING = "cl100k_base"

# --- Eval ---
RESULTS_CHECKPOINT_PATH = "results_checkpoint.jsonl"
