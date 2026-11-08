#!/usr/bin/env python3
"""Exercises the L1 exact-match cache.

It verifies the miss-to-hit transition for identical requests and confirms that
stream and non-stream variants use distinct cache keys.
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

TITLE = "03 — L1 Exact Cache"


async def main() -> None:
    c.start(TITLE, "Verify L1 exact-match cache miss/hit cycle and key isolation")
    keys = c.proxy_keys()
    key = keys[0]

    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=5.0)) as client:
        r = c.redis_client()
        try:
            scope = c.scope_for(key)
            await c.cleanup_scope(r, scope)
            c.section("First request (expect MISS)")
            body = c.make_payload("Name the smallest planet in our solar system. One word.")
            reply1 = await c.send(client, c.DEV_URL, key, body, "first")
            c.check("first call status 200", reply1.status == 200, f"status={reply1.status}")
            c.check("first call is MISS", reply1.x_cache == "MISS", reply1.x_cache)
            ctx = c.ctx_for(body, key)
            waited = await c.wait_for_key(r, ctx.exact_key, timeout=10)
            c.check("L1 key persisted", waited is not None,
                     f"waited={waited:.2f}s" if waited else "timed out")
            c.section("Inspect cached L1 key")
            facts = await c.describe_key(r, ctx.exact_key)
            c.check("L1 key type is 'string'", facts["type"] == "string", facts["type"])
            c.check("L1 key has a TTL", facts.get("ttl", 0) > 0, f"ttl={facts.get('ttl')}s")
            c.section("Second request (identical, expect HIT-EXACT)")
            reply2 = await c.send(client, c.DEV_URL, key, body, "second")
            c.check("second call status 200", reply2.status == 200, f"status={reply2.status}")
            c.check("second call is HIT-EXACT", reply2.x_cache == "HIT-EXACT", reply2.x_cache)
            c.section("Body comparison")
            c.check("HIT body matches MISS body", reply1.body == reply2.body,
                     f"miss={len(reply1.body)}B, hit={len(reply2.body)}B")
            if reply1.exec_ms and reply2.exec_ms:
                c.check("HIT is faster than MISS",
                         reply2.exec_ms < reply1.exec_ms,
                         f"miss={reply1.exec_ms:.1f}ms, hit={reply2.exec_ms:.1f}ms")
            c.section("Different prompt (expect MISS)")
            body2 = c.make_payload("Name the largest planet in our solar system. One word.")
            reply3 = await c.send(client, c.DEV_URL, key, body2, "different-prompt")
            c.check("different prompt status 200", reply3.status == 200, f"status={reply3.status}")
            c.check("different prompt is MISS", reply3.x_cache == "MISS", reply3.x_cache)
            c.section("stream=True uses a different cache key (#2)")
            body_stream = c.make_payload("Name the smallest planet in our solar system. One word.",
                                          stream=True)
            reply_stream = await c.send(client, c.DEV_URL, key, body_stream, "stream-variant")
            c.check("stream variant status 200", reply_stream.status == 200,
                     f"status={reply_stream.status}")
            # Because stream=True is part of the digest, it should be a MISS even though
            # the non-stream version is cached
            c.check("stream variant is MISS (different key)", reply_stream.x_cache == "MISS",
                     reply_stream.x_cache)
            c.section("Cleanup")
            await c.cleanup_tracked(r)

        finally:
            await r.aclose()


c.run(main)
