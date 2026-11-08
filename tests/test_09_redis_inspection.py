#!/usr/bin/env python3
"""Inspects the Redis state produced by the semantic-cache pipeline.

It verifies the exact/semantic key layout, stored fields, TTLs, and the cleanup
logic that removes test data from a scope.
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

TITLE = "09 — Redis Introspection & Cache Lifecycle"


async def main() -> None:
    c.start(TITLE, "Deep inspection of what gets stored in Redis")
    keys = c.proxy_keys()
    key = keys[0]

    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=5.0)) as client:
        r = c.redis_client()
        try:
            scope = c.scope_for(key)
            await c.cleanup_scope(r, scope)
            c.section("Seed a cacheable request")
            body = c.make_payload("What is the boiling point of water in Celsius?")
            reply = await c.send(client, c.DEV_URL, key, body, "seed")
            c.check("seed status 200", reply.status == 200, f"status={reply.status}")
            c.check("seed is MISS", reply.x_cache == "MISS", reply.x_cache)

            # Wait for background writes
            ctx = c.ctx_for(body, key)
            await c.wait_for_key(r, ctx.exact_key, timeout=10)
            await asyncio.sleep(1)  # extra time for L2
            c.section("L1 (exact) key inspection")
            l1 = await c.describe_key(r, ctx.exact_key)
            c.check("L1 exists", l1["exists"], l1["type"])
            c.check("L1 is a string", l1["type"] == "string", l1["type"])
            c.check("L1 has TTL", l1.get("ttl", 0) > 0, f"ttl={l1.get('ttl')}")
            c.check("L1 key follows naming convention",
                     ctx.exact_key.startswith(f"cache:exact:{scope}:"), ctx.exact_key)

            # Verify stored body matches the response
            stored_body = l1.get("value", b"")
            c.check("L1 body matches response", stored_body == reply.body,
                     f"stored={len(stored_body)}B, response={len(reply.body)}B")
            c.section("L2 (semantic) key inspection")
            l2 = await c.describe_key(r, ctx.semantic_key)
            c.check("L2 exists", l2["exists"], l2["type"])
            c.check("L2 is a hash", l2["type"] == "hash", l2["type"])
            c.check("L2 has TTL", l2.get("ttl", 0) > 0, f"ttl={l2.get('ttl')}")
            c.check("L2 key follows naming convention",
                     ctx.semantic_key.startswith(f"cache:sem:{scope}:"), ctx.semantic_key)

            if "hash" in l2:
                h = l2["hash"]
                field_names = sorted(k.decode() for k in h.keys())
                c.info(f"L2 hash fields: {field_names}")

                c.check("L2 has 'scope' field", b"scope" in h, field_names)
                c.check("L2 has 'fp' field", b"fp" in h, field_names)
                c.check("L2 has 'vec' field", b"vec" in h, field_names)
                c.check("L2 has 'body' field", b"body" in h, field_names)

                # Scope stored in hash matches derived scope
                stored_scope = h.get(b"scope", b"").decode()
                c.check("stored scope matches", stored_scope == scope,
                         f"stored={stored_scope}, expected={scope}")

                # FP matches
                stored_fp = h.get(b"fp", b"").decode()
                c.check("stored fp matches", stored_fp == ctx.fp,
                         f"stored={stored_fp[:16]}..., expected={ctx.fp[:16]}...")

                # Vec is 768 * 4 = 3072 bytes
                vec = h.get(b"vec", b"")
                c.check("vec is 3072 bytes (768 × float32)", len(vec) == 3072,
                         f"len={len(vec)}")

                # Body matches
                stored_body_l2 = h.get(b"body", b"")
                c.check("L2 body matches response", stored_body_l2 == reply.body,
                         f"stored={len(stored_body_l2)}B, response={len(reply.body)}B")
            c.section("All keys in scope")
            all_keys = await c.scope_keys(r, scope)
            c.info(f"Total keys in scope {scope}: {len(all_keys)}")
            for k in all_keys:
                c.detail("key", k, indent=2)
            c.check("at least 2 keys (L1 + L2)", len(all_keys) >= 2,
                     f"count={len(all_keys)}")
            c.section("Cleanup removes all test keys")
            n = await c.cleanup_scope(r, scope)
            c.info(f"cleanup_scope removed {n} keys")
            remaining = await c.scope_keys(r, scope)
            c.check("no keys remain after cleanup", len(remaining) == 0,
                     f"remaining={len(remaining)}")

        finally:
            await r.aclose()


c.run(main)
