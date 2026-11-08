"""Adversarial tests for semantic cache false-positive prevention.

Tests cover the constraint-extraction and prompt-compatibility validation
layer added to harden the L2 semantic cache against incorrect cache hits.
They also verify the full ``search_semantic`` path using mocked Redis
results, ensuring that the validation integrates correctly with distance
checks, scope/fp filtering, and Redis error handling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from redis.exceptions import RedisError

from janus.cache import (
    _Constraints,
    _escape_tag,
    extract_constraints,
    prompts_compatible,
    search_semantic,
    store,
    CacheCtx,
)


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def fake_vector(rng):
    return rng.random(768).astype(np.float32)


def _mock_redis_with_doc(
    *,
    dist: str = "0.03",
    body: bytes = b'{"ok":true}',
    prompt: str | bytes | None = None,
    docs_count: int = 1,
) -> MagicMock:
    """Build a mock Redis client whose FT().search() returns a single doc."""
    doc = MagicMock()
    doc.dist = dist
    doc.body = body
    if prompt is not None:
        doc.prompt = prompt
    else:
        # Simulate a doc missing the 'prompt' attribute entirely (legacy entry)
        doc.prompt = None
    docs = [doc] * docs_count

    # ft() is a sync method that returns an index object; search() is async.
    ft_index = MagicMock()
    result = MagicMock()
    result.docs = docs
    ft_index.search = AsyncMock(return_value=result)

    redis = MagicMock()
    redis.ft.return_value = ft_index
    return redis


def _mock_redis_empty() -> MagicMock:
    ft_index = MagicMock()
    result = MagicMock()
    result.docs = []
    ft_index.search = AsyncMock(return_value=result)
    redis = MagicMock()
    redis.ft.return_value = ft_index
    return redis


def _mock_redis_error() -> MagicMock:
    ft_index = MagicMock()
    ft_index.search = AsyncMock(side_effect=RedisError("Redis connection lost"))
    redis = MagicMock()
    redis.ft.return_value = ft_index
    return redis


# ======================================================================
# 1. extract_constraints — unit tests
# ======================================================================

class TestExtractConstraints:
    def test_extracts_integers(self):
        c = extract_constraints("give me 5 items")
        assert "5" in c.numbers

    def test_extracts_decimals(self):
        c = extract_constraints("the price is 19.99")
        assert "19.99" in c.numbers

    def test_extracts_negative_numbers(self):
        c = extract_constraints("temperature is -5 degrees")
        assert "-5" in c.numbers

    def test_extracts_ordinals(self):
        c = extract_constraints("this is the 3rd option")
        assert "3rd" in c.numbers

    def test_extracts_negation_tokens(self):
        c = extract_constraints("you cannot access this resource")
        assert "cannot" in c.negations

    def test_no_negation_in_positive_sentence(self):
        c = extract_constraints("you can access this resource")
        assert len(c.negations) == 0

    def test_extracts_quoted_entities_double(self):
        c = extract_constraints('delete user "alice"')
        assert "alice" in c.quoted_entities

    def test_extracts_quoted_entities_single(self):
        c = extract_constraints("find file 'report.pdf'")
        assert "report.pdf" in c.quoted_entities

    def test_word_set(self):
        c = extract_constraints("what is the capital of france")
        assert "capital" in c.words
        assert "france" in c.words


# ======================================================================
# 2. prompts_compatible — unit tests
# ======================================================================

class TestPromptsCompatible:
    """Test the deterministic validation layer."""

    # --- Cases that SHOULD be compatible (cache hit allowed) ---

    def test_identical_prompts(self):
        assert prompts_compatible(
            "what is the capital of france",
            "what is the capital of france",
        )

    def test_minor_wording_change(self):
        # Same meaning, same constraints — should be compatible.
        assert prompts_compatible(
            "what's the capital of france",
            "what is the capital of france",
        )

    def test_synonym_same_constraints(self):
        # Different words but no constraint violations.
        assert prompts_compatible(
            "how tall is the eiffel tower",
            "what is the height of the eiffel tower",
        )

    # --- Cases that MUST be rejected (false-positive prevention) ---

    def test_negation_change_cannot(self):
        assert not prompts_compatible(
            "can i access this",
            "can i not access this",
        )

    def test_negation_change_dont(self):
        assert not prompts_compatible(
            "do enable logging",
            "don't enable logging",
        )

    def test_opposite_actions_enable_disable(self):
        assert not prompts_compatible(
            "enable authentication",
            "disable authentication",
        )

    def test_opposite_actions_create_delete(self):
        assert not prompts_compatible(
            "create the database",
            "delete the database",
        )

    def test_opposite_actions_start_stop(self):
        assert not prompts_compatible(
            "start the service",
            "stop the service",
        )

    def test_opposite_actions_allow_deny(self):
        assert not prompts_compatible(
            "allow access to the file",
            "deny access to the file",
        )

    def test_different_numbers(self):
        assert not prompts_compatible(
            "give me 5 items",
            "give me 50 items",
        )

    def test_different_decimal_numbers(self):
        assert not prompts_compatible(
            "set the threshold to 0.5",
            "set the threshold to 0.8",
        )

    def test_different_entities_quoted(self):
        assert not prompts_compatible(
            'delete user "alice"',
            'delete user "bob"',
        )

    def test_different_currency_amounts(self):
        # Different numbers should be caught even when currency differs.
        assert not prompts_compatible(
            "what's the price 100 in gbp",
            "what's the price 200 in usd",
        )

    def test_different_dates_as_numbers(self):
        assert not prompts_compatible(
            "what happened on day 15",
            "what happened on day 16",
        )

    def test_negation_without_vs_with(self):
        assert not prompts_compatible(
            "enable the feature without restrictions",
            "enable the feature",
        )

    def test_true_false_antonym(self):
        assert not prompts_compatible(
            "set the flag to true",
            "set the flag to false",
        )

    def test_increase_decrease_antonym(self):
        assert not prompts_compatible(
            "increase the timeout",
            "decrease the timeout",
        )

    def test_before_after_antonym(self):
        assert not prompts_compatible(
            "what happened before the meeting",
            "what happened after the meeting",
        )


# ======================================================================
# 3. search_semantic — integration with prompt validation
# ======================================================================

class TestSearchSemanticValidation:
    """Test the full search_semantic path with mocked Redis."""

    @pytest.mark.anyio
    async def test_identical_query_cache_hit(self, fake_vector):
        redis = _mock_redis_with_doc(dist="0.01", prompt="what is the capital of france")
        result = await search_semantic(
            redis, "scope1", "fp1", fake_vector, 0.12,
            prompt="what is the capital of france",
        )
        assert result == b'{"ok":true}'

    @pytest.mark.anyio
    async def test_paraphrase_cache_hit(self, fake_vector):
        # Same constraints, different wording — should hit.
        redis = _mock_redis_with_doc(dist="0.05", prompt="what is the capital of france")
        result = await search_semantic(
            redis, "scope1", "fp1", fake_vector, 0.12,
            prompt="what's the capital of france",
        )
        assert result == b'{"ok":true}'

    @pytest.mark.anyio
    async def test_negation_rejected(self, fake_vector):
        redis = _mock_redis_with_doc(dist="0.03", prompt="can i access this")
        result = await search_semantic(
            redis, "scope1", "fp1", fake_vector, 0.12,
            prompt="can i not access this",
        )
        assert result is None

    @pytest.mark.anyio
    async def test_opposite_action_rejected(self, fake_vector):
        redis = _mock_redis_with_doc(dist="0.02", prompt="enable authentication")
        result = await search_semantic(
            redis, "scope1", "fp1", fake_vector, 0.12,
            prompt="disable authentication",
        )
        assert result is None

    @pytest.mark.anyio
    async def test_different_number_rejected(self, fake_vector):
        redis = _mock_redis_with_doc(dist="0.01", prompt="give me 5 items")
        result = await search_semantic(
            redis, "scope1", "fp1", fake_vector, 0.12,
            prompt="give me 50 items",
        )
        assert result is None

    @pytest.mark.anyio
    async def test_different_entity_rejected(self, fake_vector):
        redis = _mock_redis_with_doc(dist="0.01", prompt='delete user "alice"')
        result = await search_semantic(
            redis, "scope1", "fp1", fake_vector, 0.12,
            prompt='delete user "bob"',
        )
        assert result is None

    @pytest.mark.anyio
    async def test_stored_prompt_as_bytes(self, fake_vector):
        # Redis with decode_responses=False returns bytes.
        redis = _mock_redis_with_doc(
            dist="0.03",
            prompt=b"enable authentication",
        )
        result = await search_semantic(
            redis, "scope1", "fp1", fake_vector, 0.12,
            prompt="disable authentication",
        )
        assert result is None

    @pytest.mark.anyio
    async def test_legacy_entry_no_prompt_still_works(self, fake_vector):
        # Old entries without the prompt field should still match on distance
        # alone (backward compatibility).
        redis = _mock_redis_with_doc(dist="0.03", prompt=None)
        result = await search_semantic(
            redis, "scope1", "fp1", fake_vector, 0.12,
            prompt="some query text",
        )
        assert result == b'{"ok":true}'

    @pytest.mark.anyio
    async def test_no_prompt_kwarg_skips_validation(self, fake_vector):
        # When the caller does not provide `prompt=`, validation is skipped.
        redis = _mock_redis_with_doc(dist="0.03", prompt="enable authentication")
        result = await search_semantic(
            redis, "scope1", "fp1", fake_vector, 0.12,
        )
        assert result == b'{"ok":true}'


# ======================================================================
# 4. search_semantic — distance and filtering tests
# ======================================================================

class TestSearchSemanticDistanceAndFiltering:
    """Verify existing distance/filtering behaviour is preserved."""

    @pytest.mark.anyio
    async def test_within_threshold_returns_body(self, fake_vector):
        redis = _mock_redis_with_doc(dist="0.05")
        result = await search_semantic(redis, "s", "f", fake_vector, 0.12)
        assert result == b'{"ok":true}'

    @pytest.mark.anyio
    async def test_beyond_threshold_returns_none(self, fake_vector):
        redis = _mock_redis_with_doc(dist="0.20")
        result = await search_semantic(redis, "s", "f", fake_vector, 0.12)
        assert result is None

    @pytest.mark.anyio
    async def test_exact_threshold_boundary(self, fake_vector):
        # distance == max_distance should be accepted (<=).
        redis = _mock_redis_with_doc(dist="0.12")
        result = await search_semantic(redis, "s", "f", fake_vector, 0.12)
        assert result == b'{"ok":true}'

    @pytest.mark.anyio
    async def test_empty_results_returns_none(self, fake_vector):
        redis = _mock_redis_empty()
        result = await search_semantic(redis, "s", "f", fake_vector, 0.12)
        assert result is None

    @pytest.mark.anyio
    async def test_redis_error_returns_none(self, fake_vector):
        redis = _mock_redis_error()
        result = await search_semantic(redis, "s", "f", fake_vector, 0.12)
        assert result is None

    @pytest.mark.anyio
    async def test_different_scopes_isolated(self, fake_vector):
        """Different scopes should produce different TAG queries.

        While we can't test actual Redis filtering with mocks, we verify
        the query string is constructed with the correct scope.
        """
        redis = _mock_redis_with_doc(dist="0.03")
        # Call with scope_a
        await search_semantic(redis, "scope_a", "fp1", fake_vector, 0.12)
        call_args_a = redis.ft.return_value.search.call_args
        # Call with scope_b
        redis.ft.return_value.search.reset_mock()
        await search_semantic(redis, "scope_b", "fp1", fake_vector, 0.12)
        call_args_b = redis.ft.return_value.search.call_args

        # The Query objects should differ (different scope in filter).
        query_a_str = str(call_args_a[0][0].get_args())
        query_b_str = str(call_args_b[0][0].get_args())
        assert "scope_a" in query_a_str or "scope_a" in str(call_args_a)
        assert "scope_b" in query_b_str or "scope_b" in str(call_args_b)

    @pytest.mark.anyio
    async def test_different_fingerprints_isolated(self, fake_vector):
        """Different fps should produce different TAG queries."""
        redis = _mock_redis_with_doc(dist="0.03")
        await search_semantic(redis, "s", "fp_alpha", fake_vector, 0.12)
        call_a = redis.ft.return_value.search.call_args
        redis.ft.return_value.search.reset_mock()
        await search_semantic(redis, "s", "fp_beta", fake_vector, 0.12)
        call_b = redis.ft.return_value.search.call_args
        assert call_a != call_b


# ======================================================================
# 5. store — prompt persistence
# ======================================================================

class TestStorePrompt:
    """Verify that the prompt text is stored alongside the cached entry."""

    @pytest.mark.anyio
    async def test_store_includes_prompt(self, fake_vector):
        pipe = MagicMock()
        pipe.set = MagicMock()
        pipe.hset = MagicMock()
        pipe.expire = MagicMock()
        pipe.execute = AsyncMock(return_value=[])

        redis = MagicMock()
        redis.pipeline.return_value.__aenter__ = AsyncMock(return_value=pipe)
        redis.pipeline.return_value.__aexit__ = AsyncMock(return_value=False)

        ctx = CacheCtx(
            scope="s", digest="d", fp="f",
            vector=fake_vector,
            prompt="what is the capital of france",
        )
        await store(redis, ctx, b"body", 3600)

        # Check that hset was called with the prompt field.
        pipe.hset.assert_called_once()
        call_kwargs = pipe.hset.call_args
        mapping = call_kwargs[1].get("mapping") or call_kwargs[0][1]
        assert isinstance(mapping, dict)
        assert "prompt" in mapping
        assert mapping["prompt"] == "what is the capital of france"

    @pytest.mark.anyio
    async def test_store_omits_prompt_when_none(self, fake_vector):
        pipe = MagicMock()
        pipe.set = MagicMock()
        pipe.hset = MagicMock()
        pipe.expire = MagicMock()
        pipe.execute = AsyncMock(return_value=[])

        redis = MagicMock()
        redis.pipeline.return_value.__aenter__ = AsyncMock(return_value=pipe)
        redis.pipeline.return_value.__aexit__ = AsyncMock(return_value=False)

        ctx = CacheCtx(
            scope="s", digest="d", fp="f",
            vector=fake_vector,
            prompt=None,
        )
        await store(redis, ctx, b"body", 3600)

        pipe.hset.assert_called_once()
        call_kwargs = pipe.hset.call_args
        mapping = call_kwargs[1].get("mapping") or call_kwargs[0][1]
        assert isinstance(mapping, dict)
        assert "prompt" not in mapping


# ======================================================================
# 6. _escape_tag — TAG escaping
# ======================================================================

class TestEscapeTag:
    def test_hex_string_unchanged(self):
        # SHA-256 hex digests should pass through essentially unchanged.
        h = "a1b2c3d4e5f6"
        assert _escape_tag(h) == h

    def test_special_chars_escaped(self):
        assert "\\ " in _escape_tag("hello world")
        assert "\\@" in _escape_tag("user@host")

    def test_empty_string(self):
        assert _escape_tag("") == ""


# ======================================================================
# 7. Edge cases for prompts_compatible
# ======================================================================

class TestPromptsCompatibleEdgeCases:
    def test_empty_vs_empty(self):
        assert prompts_compatible("", "")

    def test_single_word(self):
        assert prompts_compatible("hello", "hello")

    def test_number_only_vs_different_number(self):
        assert not prompts_compatible("42", "43")

    def test_contraction_negation(self):
        assert not prompts_compatible(
            "i can do this",
            "i can't do this",
        )

    def test_same_numbers_different_context(self):
        # Same numbers, no negation/antonym — should be compatible
        # (cosine distance handles the semantic gap).
        assert prompts_compatible(
            "show 10 results from page 1",
            "display 10 results from page 1",
        )

    def test_currencies_same_number(self):
        # Same number, just different currency word — compatible
        # (this is a borderline case, but the number check catches
        # the important differences).
        assert prompts_compatible(
            "what's the price 100 in gbp",
            "what's the price 100 in eur",
        )

    def test_on_off_antonym(self):
        assert not prompts_compatible(
            "turn the light on",
            "turn the light off",
        )

    def test_open_close_antonym(self):
        assert not prompts_compatible(
            "open the connection",
            "close the connection",
        )

    def test_grant_revoke_antonym(self):
        assert not prompts_compatible(
            "grant admin permissions",
            "revoke admin permissions",
        )

    def test_accept_reject_antonym(self):
        assert not prompts_compatible(
            "accept the invitation",
            "reject the invitation",
        )
