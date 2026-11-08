#!/usr/bin/env python3
"""Exercises validation and edge-case handling.

The checks cover malformed payloads, TTL behavior, and scope isolation for
cache entries created under different API keys.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
os.environ.setdefault("TEST_MODEL", "nvidia/nemotron-3-ultra-550b-a55b")

import common as c  # noqa: E402
import httpx  # noqa: E402

TITLE = "08 — Edge Cases & Error Handling"


async def _raw_post(client: httpx.AsyncClient, key: str | None,
                    raw_body: bytes, label: str) -> c.Reply:
    """POST raw bytes to the endpoint and return a Reply."""
    h = c.auth_headers(key)
    t0 = time.perf_counter()
    chunks: list[tuple[float, bytes]] = []
    async with client.stream("POST", f"{c.DEV_URL}/v1/responses",
                              content=raw_body, headers=h) as resp:
        async for chunk in resp.aiter_bytes():
            chunks.append(((time.perf_counter() - t0) * 1000, chunk))
        status, hdrs = resp.status_code, resp.headers
    wall = (time.perf_counter() - t0) * 1000
    reply = c.Reply(status, hdrs, b"".join(ch for _, ch in chunks), wall, chunks)
    c.out(f"<- [{label}] status={reply.status} wall={wall:.1f}ms bytes={len(reply.body)}", 1)
    return reply


async def main() -> None:
    c.start(TITLE, "Verify error handling, validation, TTLs, and tenant isolation")
    keys = c.proxy_keys()
    key = keys[0]

    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=5.0)) as client:
        r = c.redis_client()
        try:
            c.section("Missing 'model' field → 422")
            bad1 = {"input": "hello", "temperature": 0}
            reply1 = await _raw_post(client, key, json.dumps(bad1).encode(), "no-model")
            c.check("no model → 422", reply1.status == 422, f"status={reply1.status}")
            c.section("Missing 'input' field → 422")
            bad2 = {"model": c.TEST_MODEL, "temperature": 0}
            reply2 = await _raw_post(client, key, json.dumps(bad2).encode(), "no-input")
            c.check("no input → 422", reply2.status == 422, f"status={reply2.status}")
            c.section("Invalid JSON body → 422")
            reply3 = await _raw_post(client, key, b"this is not json!", "bad-json")
            c.check("invalid JSON → 422", reply3.status == 422, f"status={reply3.status}")
            c.section("Empty body → 422")
            reply4 = await _raw_post(client, key, b"", "empty-body")
            c.check("empty body → 422", reply4.status == 422, f"status={reply4.status}")
            c.section("Extra fields (model_config extra='allow')")
            body_extra = c.make_payload("Say hi.")
            body_extra["custom_field"] = "custom_value"
            body_extra["another_extra"] = 42
            reply5 = await c.send(client, c.DEV_URL, key, body_extra, "extra-fields")
            c.check("extra fields accepted (not 422)", reply5.status == 200,
                     f"status={reply5.status}")
            c.section("TTL verification")
            body_ttl = c.make_payload("Name a primary color. One word only.")
            reply_ttl = await c.send(client, c.DEV_URL, key, body_ttl, "ttl-check")
            c.check("ttl check status 200", reply_ttl.status == 200, f"status={reply_ttl.status}")

            ctx_ttl = c.ctx_for(body_ttl, key)
            await c.wait_for_key(r, ctx_ttl.exact_key, timeout=10)
            ttl_val = await r.ttl(ctx_ttl.exact_key)
            c.check("L1 key has TTL set", ttl_val > 0, f"ttl={ttl_val}s")
            c.check("TTL is within expected range",
                     0 < ttl_val <= c.CACHE_TTL,
                     f"ttl={ttl_val}s, expected ≤ {c.CACHE_TTL}s")
            c.section("Tenant scope isolation")
            # Use the same prompt but check that caching is per-scope
            # Since .env may have only one key, we test the scope derivation logic
            scope = c.scope_for(key)
            c.info(f"Key scope: {scope}")
            all_keys = await c.scope_keys(r, scope)
            c.info(f"Keys in scope: {len(all_keys)}")
            for k in all_keys[:5]:
                c.detail("key", k)

            # Verify all keys are properly scoped
            properly_scoped = all(scope in k for k in all_keys)
            c.check("all keys contain scope hash", properly_scoped,
                     f"{len(all_keys)} keys checked")
            if len(keys) > 1:
                key_b = keys[1]
                scope_b = c.scope_for(key_b)
                c.check("different keys → different scopes",
                         scope != scope_b, f"a={scope}, b={scope_b}")

                # Same prompt with key_b should MISS even if key_a cached it
                body_iso = c.make_payload("Name a primary color. One word only.")
                reply_iso = await c.send(client, c.DEV_URL, key_b, body_iso, "tenant-b")
                c.check("key_b sees MISS (isolated scope)",
                         reply_iso.x_cache == "MISS", reply_iso.x_cache)
            else:
                c.skip("tenant isolation cross-key test (only 1 key configured)")
            c.section("Cleanup")
            await c.cleanup_tracked(r)

        finally:
            await r.aclose()


c.run(main)
