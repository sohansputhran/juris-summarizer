from .tokenization import count_tokens
from .config import CHUNK_MAX_TOKENS


def chunk_text(text: str, max_tokens: int = CHUNK_MAX_TOKENS) -> list[str]:
    """
    Split text into non-overlapping chunks, respecting paragraph
    boundaries (split on \\n\\n) via greedy packing. Falls back to a
    word-level hard split for any single paragraph that alone exceeds
    max_tokens.
    """
    paragraphs = [p for p in text.split("\n\n") if p.strip()]

    chunks = []
    current_chunk = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = count_tokens(para)

        # Fallback: a single paragraph alone exceeds the budget — hard-split it
        if para_tokens > max_tokens:
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_tokens = 0

            words = para.split()
            sub_chunk = []
            sub_tokens = 0
            for w in words:
                w_tokens = count_tokens(w + " ")
                if sub_tokens + w_tokens > max_tokens:
                    chunks.append(" ".join(sub_chunk))
                    sub_chunk = []
                    sub_tokens = 0
                sub_chunk.append(w)
                sub_tokens += w_tokens
            if sub_chunk:
                chunks.append(" ".join(sub_chunk))
            continue

        # Normal case: does this paragraph fit in the current chunk?
        if current_tokens + para_tokens > max_tokens:
            chunks.append("\n\n".join(current_chunk))
            current_chunk = [para]
            current_tokens = para_tokens
        else:
            current_chunk.append(para)
            current_tokens += para_tokens

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return chunks
