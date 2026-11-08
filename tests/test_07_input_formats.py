#!/usr/bin/env python3
"""Exercises the supported Responses API input formats.

It checks string, message-list, multi-turn, and content-part payloads while
confirming that multi-turn requests remain L1-only.
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

TITLE = "07 — Input Formats (string, message list, content parts)"


async def main() -> None:
    c.start(TITLE, "Verify all supported input formats for the Responses API")
    keys = c.proxy_keys()
    key = keys[0]

    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=5.0)) as client:
        r = c.redis_client()
        try:
            scope = c.scope_for(key)
            await c.cleanup_scope(r, scope)
            c.section("Input as a plain string")
            body_str = c.make_payload("What color is the sky on a clear day? One word.")
            reply_str = await c.send(client, c.DEV_URL, key, body_str, "string-input")
            c.check("string input status 200", reply_str.status == 200, f"status={reply_str.status}")
            answer_str = reply_str.answer()
            c.check("string input has answer", len(answer_str) > 0, c.short(answer_str, 100))
            c.section("Input as a single-message list")
            body_msg = c.make_payload([
                {
                    "role": "user",
                    "content": "What color is the sky on a clear day? One word.",
                }
            ])
            reply_msg = await c.send(client, c.DEV_URL, key, body_msg, "message-input")
            c.check("message input status 200", reply_msg.status == 200, f"status={reply_msg.status}")
            answer_msg = reply_msg.answer()
            c.check("message input has answer", len(answer_msg) > 0, c.short(answer_msg, 100))
            c.section("Input as a multi-message list")
            body_multi = c.make_payload([
                {"role": "user", "content": "My name is Alice."},
                {"role": "assistant", "content": "Hello Alice! How can I help?"},
                {"role": "user", "content": "What is my name? One word only."},
            ])
            reply_multi = await c.send(client, c.DEV_URL, key, body_multi, "multi-turn")
            c.check("multi-turn status 200", reply_multi.status == 200,
                     f"status={reply_multi.status}")
            c.check("multi-turn is MISS (cacheable)", reply_multi.x_cache == "MISS",
                     reply_multi.x_cache)
            answer_multi = reply_multi.answer()
            c.check("multi-turn has answer", len(answer_multi) > 0, c.short(answer_multi, 100))
            c.section("Multi-turn L2 eligibility check")
            await asyncio.sleep(3)
            ctx_multi = c.ctx_for(body_multi, key)
            l2_facts = await c.describe_key(r, ctx_multi.semantic_key)
            c.check("multi-turn has no L2 key (multi-message = L2 ineligible)",
                     not l2_facts["exists"], f"exists={l2_facts['exists']}")

            # But it should have an L1 key
            l1_facts = await c.describe_key(r, ctx_multi.exact_key)
            c.check("multi-turn has L1 key", l1_facts["exists"], l1_facts["type"])
            c.section("Input with content parts (input_text type)")
            body_parts = c.make_payload([
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "What is 3 + 5?"},
                        {"type": "input_text", "text": "Reply with only the number."},
                    ],
                }
            ])
            reply_parts = await c.send(client, c.DEV_URL, key, body_parts, "content-parts")
            c.check("content parts status 200", reply_parts.status == 200,
                     f"status={reply_parts.status}")
            answer_parts = reply_parts.answer()
            c.check("content parts has answer", len(answer_parts) > 0, c.short(answer_parts, 100))
            c.section("Cleanup")
            await c.cleanup_tracked(r)
            await c.cleanup_scope(r, scope)

        finally:
            await r.aclose()


c.run(main)
