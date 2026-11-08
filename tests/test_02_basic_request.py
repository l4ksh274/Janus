#!/usr/bin/env python3
"""Exercises the non-streaming Responses API path.

The checks cover a successful JSON response, expected OpenAI-compatible
fields, and the exact-cache transition from MISS to HIT-EXACT.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
os.environ.setdefault("TEST_MODEL", "nvidia/nemotron-3-ultra-550b-a55b")

import common as c  # noqa: E402
import httpx  # noqa: E402

TITLE = "02 — Basic Non-Streaming Request"


async def main() -> None:
    c.start(TITLE, "Verify round-trip JSON response and Responses API structure")
    keys = c.proxy_keys()
    key = keys[0]

    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=5.0)) as client:
        c.section("Send a cacheable non-streaming request")
        body = c.make_payload("What is 2+2? Reply with only the number.")
        reply = await c.send(client, c.DEV_URL, key, body, "first-call")
        c.check("status 200", reply.status == 200, f"status={reply.status}")
        c.check("content-type is JSON", "application/json" in reply.content_type, reply.content_type)
        c.check("X-Cache header present", reply.x_cache is not None, reply.x_cache)
        c.check("first call is MISS", reply.x_cache == "MISS", reply.x_cache)
        # Note: the proxy only sets X-Exec-Time-Ms in cached_reply(), not on
        # MISS/BYPASS responses.  We verify it below after the cache hit.
        c.check("X-Exec-Time-Ms absent on MISS (expected)", reply.exec_ms is None, reply.exec_ms)
        c.section("Validate Responses API JSON structure")
        try:
            data = reply.json()
            c.check("body is a dict", isinstance(data, dict), type(data).__name__)
            c.check("has 'id' field", "id" in data, list(data.keys())[:8])
            c.check("has 'output' field", "output" in data, list(data.keys())[:8])
            c.check("has 'status' field", "status" in data, data.get("status"))
            c.check("status is 'completed'", data.get("status") == "completed", data.get("status"))
            c.check("has 'model' field", "model" in data, data.get("model"))
            output = data.get("output", [])
            c.check("output is a non-empty list", isinstance(output, list) and len(output) > 0,
                     f"len={len(output) if isinstance(output, list) else 'N/A'}")

            if output:
                first_item = output[0]
                c.check("first output item has 'content'", "content" in first_item,
                         list(first_item.keys()) if isinstance(first_item, dict) else first_item)
                content = first_item.get("content", [])
                if content:
                    first_content = content[0]
                    c.check("content item has 'text'", "text" in first_content,
                             list(first_content.keys()) if isinstance(first_content, dict) else first_content)
                    text = first_content.get("text", "")
                    c.check("text is non-empty", len(text) > 0, c.short(text, 100))
                    c.info(f"Model answer: {c.short(text, 200)}")

        except Exception as exc:
            c.check("body is valid JSON", False, str(exc))
        c.section("Answer extraction helper")
        answer = reply.answer()
        c.check("answer() returns non-empty string", len(answer) > 0, c.short(answer, 100))
        c.section("Wait for background cache write")
        r = c.redis_client()
        try:
            ctx = c.ctx_for(body, key)
            waited = await c.wait_for_key(r, ctx.exact_key, timeout=10)
            c.check("L1 key written to Redis", waited is not None,
                     f"waited={waited:.2f}s" if waited else "timed out")
        finally:
            await r.aclose()
        c.section("Cache hit carries X-Exec-Time-Ms")
        reply_hit = await c.send(client, c.DEV_URL, key, body, "cached-hit")
        c.check("cache hit status 200", reply_hit.status == 200, f"status={reply_hit.status}")
        c.check("cache hit is HIT-EXACT", reply_hit.x_cache == "HIT-EXACT", reply_hit.x_cache)
        c.check("X-Exec-Time-Ms present on HIT", reply_hit.exec_ms is not None, reply_hit.exec_ms)
        if reply_hit.exec_ms is not None:
            c.check("X-Exec-Time-Ms > 0", reply_hit.exec_ms > 0, f"{reply_hit.exec_ms:.2f}ms")
        c.section("Cleanup")
        r = c.redis_client()
        try:
            await c.cleanup_tracked(r)
        finally:
            await r.aclose()


c.run(main)
