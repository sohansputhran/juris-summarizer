import tiktoken

from .config import TIKTOKEN_ENCODING

# Cached once at import time rather than re-created inside count_tokens()
# on every call — same behavior as your notebook version, just faster
# when called thousands of times across chunking + batch eval.
_enc = tiktoken.get_encoding(TIKTOKEN_ENCODING)


def count_tokens(text: str) -> int:
    return len(_enc.encode(text))
