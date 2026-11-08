"""Redis-backed exact and semantic cache helpers for Janus responses.

The module assumes a Redis client with `decode_responses=False` so cached
responses remain raw bytes and can be replayed without character-decoding
errors.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray
from redis.asyncio import Redis
from redis.commands.search.field import TagField, VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from redis.exceptions import RedisError, ResponseError

from janus import embeddings

logger = logging.getLogger(__name__)

INDEX_NAME: Final = "idx:cache:semantic"
SEMANTIC_PREFIX: Final = "cache:sem:"

_NON_TEXT_TYPES: Final = frozenset({"input_file", "input_image", "input_audio"})

# ---------------------------------------------------------------------------
# Prompt-level constraint extraction for false-positive prevention
# ---------------------------------------------------------------------------

# Tokens whose presence or absence flips the intent of a request.
_NEGATION_TOKENS: Final = frozenset({
    "not", "no", "never", "none", "neither", "nor", "without",
    "don't", "dont", "doesn't", "doesnt", "didn't", "didnt",
    "won't", "wont", "wouldn't", "wouldnt", "can't", "cant",
    "cannot", "couldn't", "couldnt", "shouldn't", "shouldnt",
    "isn't", "isnt", "aren't", "arent", "wasn't", "wasnt",
    "weren't", "werent", "hasn't", "hasnt", "haven't", "havent",
    "hadn't", "hadnt", "disable", "remove", "revoke", "deny",
    "exclude", "reject", "unlike", "unable",
})

# Antonym pairs where swapping one for the other changes meaning.
_ANTONYM_PAIRS: Final = frozenset({
    ("enable", "disable"),
    ("add", "remove"),
    ("create", "delete"),
    ("grant", "revoke"),
    ("allow", "deny"),
    ("include", "exclude"),
    ("accept", "reject"),
    ("start", "stop"),
    ("open", "close"),
    ("activate", "deactivate"),
    ("increase", "decrease"),
    ("before", "after"),
    ("above", "below"),
    ("true", "false"),
    ("yes", "no"),
    ("on", "off"),
})

# Pre-compute a lookup from each word in any antonym pair to its opposite(s).
_ANTONYM_MAP: dict[str, set[str]] = {}
for _a, _b in _ANTONYM_PAIRS:
    _ANTONYM_MAP.setdefault(_a, set()).add(_b)
    _ANTONYM_MAP.setdefault(_b, set()).add(_a)

# Pattern that matches numbers (integers, decimals, ordinals) in text.
_NUMBER_RE: Final = re.compile(
    r"""
    (?<!\w)          # not preceded by a word char (avoid matching inside words)
    -?               # optional negative sign
    \d[\d,]*         # integer part (with optional comma grouping)
    (?:\.\d+)?       # optional decimal part
    (?:st|nd|rd|th)? # optional ordinal suffix
    (?!\w)           # not followed by a word char
    """,
    re.VERBOSE,
)

# Pattern that matches quoted strings (single or double) — treated as entities.
_QUOTED_RE: Final = re.compile(r"""(?:"([^"]+)"|'([^']+)')""")

# Simple word tokenizer.
_WORD_RE: Final = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


@dataclass(frozen=True, slots=True)
class _Constraints:
    """Deterministic constraints extracted from a prompt for validation."""

    numbers: frozenset[str]
    negations: frozenset[str]
    quoted_entities: frozenset[str]
    words: frozenset[str]


def extract_constraints(text: str) -> _Constraints:
    """Extract deterministic constraint tokens from a normalised prompt.

    This function is intentionally fast and avoids NLP dependencies.  It
    captures the features most likely to distinguish semantically similar
    but functionally different requests: numbers, negation markers, quoted
    entity names, and the full word set.
    """
    lower = text.lower()
    numbers = frozenset(_NUMBER_RE.findall(lower))
    words = frozenset(_WORD_RE.findall(lower))
    negations = words & _NEGATION_TOKENS
    quoted = frozenset(
        m.group(1) or m.group(2) for m in _QUOTED_RE.finditer(lower)
    )
    return _Constraints(
        numbers=numbers,
        negations=negations,
        quoted_entities=quoted,
        words=words,
    )


def prompts_compatible(incoming: str, cached: str) -> bool:
    """Return True when *incoming* is safe to serve from the *cached* response.

    The check is deliberately conservative: any detected difference in
    numbers, negation polarity, antonym usage, or quoted entities causes a
    rejection.  The caller should treat a ``False`` return as a cache miss,
    not an error.
    """
    # Fast path: identical normalised text is always compatible.
    if incoming == cached:
        return True

    inc = extract_constraints(incoming)
    cac = extract_constraints(cached)

    # 1. Number mismatch — "5 items" vs "50 items".
    if inc.numbers != cac.numbers:
        logger.debug("prompt rejected: numbers differ inc=%s cached=%s",
                      inc.numbers, cac.numbers)
        return False

    # 2. Negation polarity change — "can access" vs "cannot access".
    if inc.negations != cac.negations:
        logger.debug("prompt rejected: negation differs inc=%s cached=%s",
                      inc.negations, cac.negations)
        return False

    # 3. Antonym swap — "enable auth" vs "disable auth".
    inc_only = inc.words - cac.words
    cac_only = cac.words - inc.words
    for word in inc_only:
        antonyms = _ANTONYM_MAP.get(word)
        if antonyms and antonyms & cac_only:
            logger.debug("prompt rejected: antonym swap %r ↔ %s", word,
                          antonyms & cac_only)
            return False

    # 4. Quoted entity mismatch — 'user "A"' vs 'user "B"'.
    if inc.quoted_entities != cac.quoted_entities:
        # Only reject when both have quoted entities (one having none is ok —
        # the cosine distance should catch the semantic gap).
        if inc.quoted_entities and cac.quoted_entities:
            logger.debug("prompt rejected: quoted entities differ inc=%s cached=%s",
                          inc.quoted_entities, cac.quoted_entities)
            return False

    return True


def _escape_tag(value: str) -> str:
    """Escape special characters for safe interpolation into a RediSearch TAG filter.

    TAG queries in RediSearch treat many punctuation characters as syntax.
    Since ``scope`` and ``fp`` are currently hex SHA-256 digests this is a
    defense-in-depth measure against future changes to those generators.
    """
    for ch in r"\,.<>{}[]\"':;!@#$%^&*()-+=~/ ":
        value = value.replace(ch, f"\\{ch}")
    return value


# ---------------------------------------------------------------------------
# CacheCtx — per-request cache metadata
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class CacheCtx:
    """Tracks the cache keys and optional semantic vector for one request."""

    scope: str
    digest: str
    fp: str
    vector: NDArray[np.float32] | None = None
    prompt: str | None = field(default=None, repr=False)

    @classmethod
    def build(cls, payload: dict[str, Any], scope: str) -> "CacheCtx":
        """Create a request-scoped cache context from a proxied payload."""
        return cls(
            scope=scope,
            digest=embeddings.request_digest(payload),
            fp=embeddings.params_fingerprint(payload),
        )

    @property
    def exact_key(self) -> str:
        """Return the Redis key used by the exact-match cache."""
        return f"cache:exact:{self.scope}:{self.digest}"

    @property
    def semantic_key(self) -> str:
        """Return the Redis hash key used for semantic candidate storage."""
        return f"{SEMANTIC_PREFIX}{self.scope}:{self.digest}"


# ---------------------------------------------------------------------------
# Cacheability checks
# ---------------------------------------------------------------------------

def _has_non_text(node: Any) -> bool:
    """Return True when the input tree contains non-text payload types."""
    if isinstance(node, dict):
        if node.get("type") in _NON_TEXT_TYPES:
            return True
        return any(_has_non_text(v) for v in node.values())
    if isinstance(node, list):
        return any(_has_non_text(v) for v in node)
    return False


def is_cacheable(payload: dict[str, Any]) -> bool:
    """Return whether a Responses API payload is safe to cache.

    Exact and semantic caches are only used for deterministic text-only prompts
    that do not depend on caller state or tool execution.
    """
    if payload.get("temperature") != 0:
        return False
    if payload.get("previous_response_id") or payload.get("background"):
        return False
    if payload.get("tools"):
        return False
    return not _has_non_text(payload.get("input"))


def is_cacheable_completion(payload: dict[str, Any]) -> bool:
    """Return whether a chat-completions payload matches the text-only cache policy."""
    if payload.get("temperature") != 0:
        return False
    if payload.get("tools") or payload.get("tool_choice"):
        return False
    for msg in payload.get("messages", []):
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") not in (None, "text"):
                    return False
    return True


# ---------------------------------------------------------------------------
# Redis index management
# ---------------------------------------------------------------------------

async def ensure_index(redis: Redis) -> bool:
    """Create the RediSearch semantic index when it is missing.

    Returns False when Redis Stack is unavailable or the index cannot be created.
    """
    try:
        await redis.ft(INDEX_NAME).create_index(
            fields=[
                TagField("scope"),
                TagField("fp"),
                VectorField(
                    "vec",
                    "HNSW",
                    {
                        "TYPE": "FLOAT32",
                        "DIM": embeddings.EMBED_DIM,
                        "DISTANCE_METRIC": "COSINE",
                    },
                ),
            ],
            definition=IndexDefinition(prefix=[SEMANTIC_PREFIX], index_type=IndexType.HASH),
        )
    except ResponseError as exc:
        if "already exists" in str(exc).lower():
            return True
        logger.warning("Could not create vector index (is this Redis Stack?): %s", exc)
        return False
    except RedisError:
        logger.warning("Redis error creating vector index", exc_info=True)
        return False
    return True


# ---------------------------------------------------------------------------
# Cache read
# ---------------------------------------------------------------------------

async def get_exact(redis: Redis, key: str) -> bytes | None:
    """Read an exact-match cache entry and return its raw response body."""
    try:
        value = await redis.get(key)
        return value.encode() if isinstance(value, str) else value
    except RedisError:
        logger.warning("Redis GET failed; bypassing cache", exc_info=True)
        return None


async def search_semantic(
    redis: Redis,
    scope: str,
    fp: str,
    vector: NDArray[np.float32],
    max_distance: float,
    *,
    prompt: str | None = None,
) -> bytes | None:
    """Return the nearest cached body for the same scope and parameter set.

    When *prompt* is provided the function performs an additional
    deterministic compatibility check between the incoming prompt and the
    prompt stored alongside the cached entry.  This prevents false-positive
    hits caused by queries that are semantically similar (low cosine
    distance) but functionally different (e.g. negation, different numbers,
    different entities).

    Entries written before the ``prompt`` field was introduced are still
    eligible — the validation step is skipped when the stored prompt is
    absent, falling back to distance-only matching.
    """
    safe_scope = _escape_tag(scope)
    safe_fp = _escape_tag(fp)
    q = (
        Query(
            "(@scope:{%s} @fp:{%s})=>[KNN 1 @vec $vec AS dist]"
            % (safe_scope, safe_fp)
        )
        .sort_by("dist")
        .return_fields("body", "dist", "prompt")
        .dialect(2)
    )
    try:
        result = await redis.ft(INDEX_NAME).search(
            q, query_params={"vec": vector.tobytes()}
        )
    except RedisError:
        logger.warning("Redis vector search failed; bypassing L2", exc_info=True)
        return None

    docs = getattr(result, "docs", None)
    if not docs:
        return None

    doc = docs[0]
    distance = float(doc.dist)
    logger.debug(
        "L2 nearest distance=%.4f (max %.4f)",
        distance,
        max_distance,
    )
    if distance > max_distance:
        return None

    # --- Prompt compatibility validation (new) --------------------------
    if prompt is not None:
        stored_prompt = getattr(doc, "prompt", None)
        # Stored prompt may be bytes (decode_responses=False) or str.
        if isinstance(stored_prompt, bytes):
            stored_prompt = stored_prompt.decode("utf-8", "replace")
        if stored_prompt and not prompts_compatible(prompt, stored_prompt):
            logger.info(
                "L2 candidate rejected by prompt validation "
                "(distance=%.4f, incoming=%r, cached=%r)",
                distance,
                prompt[:80],
                stored_prompt[:80],
            )
            return None

    return doc.body


# ---------------------------------------------------------------------------
# Cache write
# ---------------------------------------------------------------------------

async def store(
    redis: Redis,
    ctx: CacheCtx,
    body: bytes,
    ttl: int,
    *,
    exact_enabled: bool = True,
    semantic_enabled: bool = True,
) -> None:
    """Write enabled cache entries and, when present, the semantic vector payload."""
    try:
        async with redis.pipeline(transaction=False) as pipe:
            if exact_enabled:
                pipe.set(ctx.exact_key, body, ex=ttl)
            if semantic_enabled and ctx.vector is not None:
                mapping: dict[str, Any] = {
                    "scope": ctx.scope,
                    "fp": ctx.fp,
                    "vec": ctx.vector.tobytes(),
                    "body": body,
                }
                if ctx.prompt is not None:
                    mapping["prompt"] = ctx.prompt
                pipe.hset(ctx.semantic_key, mapping=mapping)  # type: ignore[arg-type]
                pipe.expire(ctx.semantic_key, ttl)
            await pipe.execute()
    except RedisError:
        logger.warning("Redis write failed; response not cached", exc_info=True)
