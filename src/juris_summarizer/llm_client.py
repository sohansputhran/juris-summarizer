import os

from dotenv import load_dotenv

from .config import (
    MAP_PROVIDER,
    REDUCE_PROVIDER,
    MAP_MODEL,
    REDUCE_MODEL,
    MAP_MAX_TOKENS,
    REDUCE_MAX_TOKENS,
    TEMPERATURE,
    REASONING_EFFORT,
)
from .prompts import MAP_PROMPT_TEMPLATE, REDUCE_PROMPT_TEMPLATE, format_chunk_summaries

load_dotenv()


def get_client(provider: str):
    """Returns an OpenAI-SDK-compatible client for the given provider ('groq' or 'cerebras')."""
    if provider == "groq":
        from groq import Groq
        return Groq(api_key=os.environ.get("GROQ_API_KEY"))
    
    elif provider == "cerebras":
        from cerebras.cloud.sdk import Cerebras
        return Cerebras(api_key=os.environ.get("CEREBRAS_API_KEY"))
    
    elif provider == "huggingface":
        from huggingface_hub import InferenceClient
        return InferenceClient(api_key=os.environ.get("HF_TOKEN"))
    
    else:
        raise ValueError(f"Unknown provider: {provider}")


def _check_finish_reason(response, context: str = ""):
    finish_reason = response.choices[0].finish_reason
    if finish_reason == "length":
        print(f"WARNING: truncated output (finish_reason=length) {context}")
    return finish_reason


def summarize_chunk(client, chunk_text: str, chunk_num: int, total_chunks: int) -> str:
    prompt = MAP_PROMPT_TEMPLATE.format(
        chunk_num=chunk_num,
        total_chunks=total_chunks,
        chunk_text=chunk_text,
    )

    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=MAP_MODEL,
        temperature=TEMPERATURE,
        max_tokens=MAP_MAX_TOKENS,
        reasoning_effort=REASONING_EFFORT,
    )

    _check_finish_reason(response, context=f"(map, chunk {chunk_num}/{total_chunks})")

    content = response.choices[0].message.content
    if not content or not content.strip():
        print(f"WARNING: empty summary for chunk {chunk_num}/{total_chunks}")

    return content


def reduce_summaries(
    client,
    chunk_summaries: list[str],
    example_chunk_summaries: str,
    example_reference_summary: str,
) -> str:
    combined = format_chunk_summaries(chunk_summaries)

    prompt = REDUCE_PROMPT_TEMPLATE.format(
        example_chunk_summaries=example_chunk_summaries,
        example_reference_summary=example_reference_summary,
        combined_summaries=combined,
    )

    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=REDUCE_MODEL,
        temperature=TEMPERATURE,
        max_tokens=REDUCE_MAX_TOKENS,
    )

    _check_finish_reason(response, context="(reduce)")

    return response.choices[0].message.content
