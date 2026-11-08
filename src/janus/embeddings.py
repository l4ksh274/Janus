"""Helpers for hashing request payloads and building semantic cache vectors.

The module keeps a stable request fingerprint for exact-cache checks, strips
volatile fields before hashing, and extracts a single text string that can be
embedded for L2 matching.
"""

import hashlib
import json
import re
from typing import Any, Final

import numpy as np
from fastembed import TextEmbedding
from numpy.typing import NDArray

EMBED_MODEL_NAME: Final = "BAAI/bge-base-en-v1.5"
EMBED_DIM: Final = 768
MAX_SEMANTIC_CHARS: Final = 1000

# Only fields that change runtime state without affecting the model response are
# excluded from cache keys. Everything else is part of the fingerprint.
VOLATILE_FIELDS: Final = frozenset({"user", "metadata"})


def _canonical(obj: Any) -> str:
    """Serialize a Python object into a stable JSON string for hashing."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _sha256(text: str) -> str:
    """Return the SHA-256 hex digest for a UTF-8 string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def request_digest(payload: dict[str, Any]) -> str:
    """Return a stable hash of the entire request, excluding volatile fields.

    The digest includes streaming mode and the request body itself so exact cache
    entries are distinct from semantically similar prompts with different
    parameters.
    """
    stable = {k: v for k, v in payload.items() if k not in VOLATILE_FIELDS}
    return _sha256(_canonical(stable))


_CONVERSATION_FIELDS = {"input", "messages"}
_INSTRUCTION_ROLES = {"system", "developer"}

def params_fingerprint(payload: dict[str, Any]) -> str:
    """Return the fingerprint for a request body, excluding conversational input.

    `input` (Responses) and non-instruction `messages` (Chat Completions) are
    left out so L2 can match semantically similar prompts. System/developer
    messages stay in, since they play the role of `instructions`.
    """
    stable = {
        k: v
        for k, v in payload.items()
        if k not in VOLATILE_FIELDS and k not in _CONVERSATION_FIELDS
    }
    messages = payload.get("messages")
    if messages:
        stable["_instructions"] = [
            m for m in messages if m.get("role") in _INSTRUCTION_ROLES
        ]
    return _sha256(_canonical(stable))


def semantic_text(raw_input: str | list[dict[str, Any]]) -> str | None:
    """Return text suitable for semantic caching, or None when the prompt is ineligible.

    Only short, single-turn, text-only user inputs are eligible. Long or
    multi-turn payloads are excluded because they can generate false-positive
    semantic hits or embed noisy JSON syntax.
    """
    if isinstance(raw_input, str):
        text = raw_input
    else:
        if len(raw_input) != 1:
            return None
        msg = raw_input[0]
        if msg.get("type", "message") != "message" or msg.get("role", "user") != "user":
            return None
        content = msg.get("content")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if not isinstance(part, dict) or part.get("type") != "input_text":
                    return None
                parts.append(str(part.get("text", "")))
            text = " ".join(parts)
        else:
            return None

    text = re.sub(r"\s+", " ", text.strip().lower())
    if not text or len(text) > MAX_SEMANTIC_CHARS:
        return None
    return text


def generate_vector(model: TextEmbedding, text: str) -> NDArray[np.float32]:
    """Embed a single prompt into the model's normalized vector space."""
    raw_vec = next(iter(model.embed([text])))
    return np.asarray(raw_vec, dtype=np.float32)


def semantic_text_from_messages(messages: list[dict]) -> str | None:
    """Extract the single user message text that may be used for L2 matching.

    Matching rules mirror `semantic_text`; the function ignores multi-message,
    multi-modal, or overlong requests.
    """
    user_msgs = [m for m in messages if m.get("role") == "user"]
    if len(user_msgs) != 1:
        return None
    content = user_msgs[0].get("content")
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if not isinstance(part, dict) or part.get("type") != "text":
                return None
            parts.append(str(part.get("text", "")))
        text = " ".join(parts)
    else:
        return None

    text = re.sub(r"\s+", " ", text.strip().lower())
    if not text or len(text) > MAX_SEMANTIC_CHARS:
        return None
    return text
