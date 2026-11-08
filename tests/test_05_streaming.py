#!/usr/bin/env python3
"""Exercises the SSE response path.

It verifies the event stream format, reconstructs a response from deltas, and
checks that cached stream replays match the original payload.
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

TITLE = "05 — Streaming Responses"


async def main() -> None:
    c.start(TITLE, "Verify SSE streaming, event structure, and stream caching")
    keys = c.proxy_keys()
    key = keys[0]

    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=5.0)) as client:
        r = c.redis_client()
        try:
            scope = c.scope_for(key)
            await c.cleanup_scope(r, scope)
            c.section("Streaming request")
            body = c.make_payload("What is 7 times 8? Reply with only the number.",
                                   stream=True)
            reply1 = await c.send(client, c.DEV_URL, key, body, "stream-first")
            c.check("stream status 200", reply1.status == 200, f"status={reply1.status}")
            c.check("content-type is event-stream",
                     "text/event-stream" in reply1.content_type, reply1.content_type)
            c.check("stream is MISS", reply1.x_cache == "MISS", reply1.x_cache)
            c.section("SSE event structure")
            events = reply1.events
            c.check("at least one event parsed", len(events) > 0, f"event_count={len(events)}")

            event_names = [name for name, _ in events]
            c.info(f"Event types: {event_names}")

            has_delta = any(n == "response.output_text.delta" for n in event_names)
            has_completed = any(n == "response.completed" for n in event_names)
            c.check("has response.output_text.delta events", has_delta, event_names)
            c.check("has response.completed event", has_completed, event_names)
            c.section("Reconstruct answer from deltas")
            answer = reply1.answer()
            c.check("answer is non-empty", len(answer) > 0, c.short(answer, 100))
            c.info(f"Streamed answer: {c.short(answer, 200)}")
            if reply1.ttfb_ms is not None:
                c.check("TTFB measured", reply1.ttfb_ms > 0, f"ttfb={reply1.ttfb_ms:.1f}ms")
                c.info(f"TTFB: {reply1.ttfb_ms:.1f}ms, total: {reply1.wall_ms:.1f}ms")
            c.section("Stream caching")
            ctx = c.ctx_for(body, key)
            waited = await c.wait_for_key(r, ctx.exact_key, timeout=15)
            c.check("stream L1 key written", waited is not None,
                     f"waited={waited:.2f}s" if waited else "timed out")

            if waited is not None:
                facts = await c.describe_key(r, ctx.exact_key)
                c.check("cached stream is a string", facts["type"] == "string", facts["type"])
            c.section("Replay cached stream")
            reply2 = await c.send(client, c.DEV_URL, key, body, "stream-cached")
            c.check("replay status 200", reply2.status == 200, f"status={reply2.status}")
            c.check("replay is HIT-EXACT", reply2.x_cache == "HIT-EXACT", reply2.x_cache)
            c.check("replay content-type is event-stream",
                     "text/event-stream" in reply2.content_type, reply2.content_type)

            # Body should be the SSE payload
            c.check("replay body matches original", reply1.body == reply2.body,
                     f"orig={len(reply1.body)}B, replay={len(reply2.body)}B")
            c.section("Chunk analysis")
            c.check("first call received multiple chunks",
                     len(reply1.chunks) > 1,
                     f"chunks={len(reply1.chunks)}")
            if reply1.chunks:
                c.info(f"Total chunks: {len(reply1.chunks)}, "
                       f"total bytes: {sum(len(ch) for _, ch in reply1.chunks)}")
            c.section("Cleanup")
            await c.cleanup_tracked(r)
            await c.cleanup_scope(r, scope)

        finally:
            await r.aclose()


c.run(main)
