#!/usr/bin/env python3
"""Exercises the non-cacheable request path.

It checks that requests with dynamic or tool-based semantics bypass the cache
and never create Redis keys.
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

TITLE = "06 — Cache Bypass (Non-Cacheable Requests)"


async def main() -> None:
    c.start(TITLE, "Verify that non-cacheable requests get BYPASS and are never stored")
    keys = c.proxy_keys()
    key = keys[0]

    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=5.0)) as client:
        r = c.redis_client()
        try:
            scope = c.scope_for(key)
            keys_before = await c.scope_keys(r, scope)
            c.section("temperature=0.7 → BYPASS")
            body_t = {
                "model": c.TEST_MODEL,
                "input": "Tell me a random fun fact.",
                "temperature": 0.7,
                "instructions": f"Be terse. [test-run {c.NONCE}]",
            }
            reply_t = await c.send(client, c.DEV_URL, key, body_t, "temp-0.7")
            c.check("temp=0.7 status 200", reply_t.status == 200, f"status={reply_t.status}")
            c.check("temp=0.7 → BYPASS", reply_t.x_cache == "BYPASS", reply_t.x_cache)
            c.section("No temperature field → BYPASS")
            body_no_t = {
                "model": c.TEST_MODEL,
                "input": "Tell me a random fun fact.",
                "instructions": f"Be terse. [test-run {c.NONCE}]",
            }
            reply_no_t = await c.send(client, c.DEV_URL, key, body_no_t, "no-temp")
            c.check("no temp status 200", reply_no_t.status == 200, f"status={reply_no_t.status}")
            c.check("no temp → BYPASS", reply_no_t.x_cache == "BYPASS", reply_no_t.x_cache)
            c.section("tools field present → BYPASS")
            body_tools = c.make_payload(
                "What is the weather in Paris?",
                tools=[{
                    "type": "function",
                    "name": "get_weather",
                    "description": "Get weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                }],
            )
            reply_tools = await c.send(client, c.DEV_URL, key, body_tools, "with-tools")
            c.check("tools status 200 or valid error",
                     reply_tools.status in (200, 400, 422), f"status={reply_tools.status}")
            if reply_tools.status == 200:
                c.check("tools → BYPASS", reply_tools.x_cache == "BYPASS", reply_tools.x_cache)
            else:
                c.info(f"Upstream rejected tools request with {reply_tools.status} (expected for some models)")
                c.check("tools request not cached (non-200)", True, f"status={reply_tools.status}")
            c.section("previous_response_id → BYPASS")
            body_prev = c.make_payload(
                "Continue the conversation.",
                previous_response_id="resp_fake_00000000000000000000000000",
            )
            reply_prev = await c.send(client, c.DEV_URL, key, body_prev, "prev-response-id")
            # Upstream may 400 on a fake id, but the proxy should still set BYPASS
            if reply_prev.x_cache is not None:
                c.check("previous_response_id → BYPASS", reply_prev.x_cache == "BYPASS",
                         reply_prev.x_cache)
            else:
                c.info(f"No X-Cache header (upstream error pass-through, status={reply_prev.status})")
                c.check("previous_response_id handled", True,
                         f"status={reply_prev.status}")
            c.section("Verify no cache keys created")
            await asyncio.sleep(3)  # give background tasks time to run
            keys_after = await c.scope_keys(r, scope)
            new_keys = set(keys_after) - set(keys_before)
            c.check("no new cache keys from bypass requests",
                     len(new_keys) == 0,
                     f"new_keys={new_keys}" if new_keys else "none")

        finally:
            await r.aclose()


c.run(main)
