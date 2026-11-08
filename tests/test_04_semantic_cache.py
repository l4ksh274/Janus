#!/usr/bin/env python3
"""Exercises the L2 semantic-cache checks.

The script seeds the cache with a short prompt and then validates paraphrase
matching, ineligible long prompts, and parameter-fingerprint isolation.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
os.environ.setdefault("TEST_MODEL", "nvidia/nemotron-3-ultra-550b-a55b")

import common as c  # noqa: E402
import httpx  # noqa: E402

TITLE = "04 — L2 Semantic Cache"


async def main() -> None:
    c.start(TITLE, "Verify L2 semantic-match cache for paraphrased prompts")
    keys = c.proxy_keys()
    key = keys[0]

    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=5.0)) as client:
        r = c.redis_client()
        try:
            scope = c.scope_for(key)
            await c.cleanup_scope(r, scope)
            c.section("Seed request (expect MISS)")
            body1 = c.make_payload("What is the capital of France?")
            reply1 = await c.send(client, c.DEV_URL, key, body1, "seed")
            c.check("seed status 200", reply1.status == 200, f"status={reply1.status}")
            c.check("seed is MISS", reply1.x_cache == "MISS", reply1.x_cache)

            # Wait for both L1 + L2 to be written
            ctx1 = c.ctx_for(body1, key)
            await c.wait_for_key(r, ctx1.exact_key, timeout=10)
            await c.wait_for_key(r, ctx1.semantic_key, timeout=10)
            c.section("Verify L2 key exists in Redis")
            l2_facts = await c.describe_key(r, ctx1.semantic_key)
            c.check("L2 key exists", l2_facts["exists"], l2_facts["type"])
            c.check("L2 key is a hash", l2_facts["type"] == "hash", l2_facts["type"])
            if "hash" in l2_facts:
                c.check("L2 hash has 'vec' field", b"vec" in l2_facts["hash"],
                         sorted(k.decode() for k in l2_facts["hash"].keys()))
                c.check("L2 hash has 'body' field", b"body" in l2_facts["hash"],
                         sorted(k.decode() for k in l2_facts["hash"].keys()))
                vec_bytes = l2_facts["hash"].get(b"vec", b"")
                c.check("vec is 768 float32s (1536 bytes)", len(vec_bytes) == 768 * 4,
                         f"vec_bytes={len(vec_bytes)}")
            c.section("Paraphrased prompt (expect HIT-SEMANTIC)")
            # The default SEMANTIC_MAX_DISTANCE is 0.08 (very tight), so the
            # paraphrase must be extremely close to the seed.
            body2 = c.make_payload("What's the capital of France?")
            reply2 = await c.send(client, c.DEV_URL, key, body2, "paraphrase")
            c.check("paraphrase status 200", reply2.status == 200, f"status={reply2.status}")
            c.check("paraphrase is HIT-SEMANTIC", reply2.x_cache == "HIT-SEMANTIC", reply2.x_cache)

            # The body from a semantic hit should match the seed's body
            if reply2.x_cache == "HIT-SEMANTIC":
                c.check("semantic hit body matches seed",
                         reply2.body == reply1.body,
                         f"seed={len(reply1.body)}B, hit={len(reply2.body)}B")
            c.section("Completely different prompt (expect MISS)")
            body3 = c.make_payload("How do you sort a list in Python?")
            reply3 = await c.send(client, c.DEV_URL, key, body3, "different")
            c.check("different prompt status 200", reply3.status == 200, f"status={reply3.status}")
            c.check("different prompt is MISS", reply3.x_cache == "MISS", reply3.x_cache)
            c.section("Same prompt, different instructions (params fingerprint differs)")
            body4 = c.make_payload("What is the capital of France?",
                                    instructions=f"Be very verbose and detailed. [test-run {c.NONCE}]")
            reply4 = await c.send(client, c.DEV_URL, key, body4, "diff-instructions")
            c.check("different instructions status 200", reply4.status == 200,
                     f"status={reply4.status}")
            c.check("different instructions is MISS (fp differs)",
                     reply4.x_cache == "MISS", reply4.x_cache)
            c.section("Long prompt (>1000 chars, L2 ineligible)")
            long_text = "Explain quantum mechanics. " * 100  # ~2700 chars
            body5 = c.make_payload(long_text)
            reply5 = await c.send(client, c.DEV_URL, key, body5, "long-prompt")
            c.check("long prompt status 200", reply5.status == 200, f"status={reply5.status}")
            # L2 won't be written for this; verify by checking that after waiting,
            # the semantic key does NOT exist
            await asyncio.sleep(3)
            ctx5 = c.ctx_for(body5, key)
            l2_long = await c.describe_key(r, ctx5.semantic_key)
            c.check("long prompt has no L2 key", not l2_long["exists"],
                     f"exists={l2_long['exists']}")
            c.section("Cleanup")
            await c.cleanup_tracked(r)
            await c.cleanup_scope(r, scope)

        finally:
            await r.aclose()


c.run(main)
