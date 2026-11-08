#!/usr/bin/env python3
"""Measures cache performance and timing behavior.

This script compares cache misses to hits for exact and semantic paths and
verifies the timing headers for non-streaming and streaming responses.
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

TITLE = "10 — Performance & Timing"


async def main() -> None:
    c.start(TITLE, "Measure cache speedup and timing consistency")
    keys = c.proxy_keys()
    key = keys[0]

    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=5.0)) as client:
        r = c.redis_client()
        try:
            scope = c.scope_for(key)
            await c.cleanup_scope(r, scope)
            c.section("Seed request (non-streaming)")
            body = c.make_payload("What is the speed of light in km/s? Just the number.")
            reply_miss = await c.send(client, c.DEV_URL, key, body, "miss")
            c.check("seed status 200", reply_miss.status == 200, f"status={reply_miss.status}")
            miss_ms = reply_miss.wall_ms

            ctx = c.ctx_for(body, key)
            await c.wait_for_key(r, ctx.exact_key, timeout=10)
            c.section("Multiple cache hits (non-streaming)")
            hit_times: list[float] = []
            for i in range(5):
                reply_hit = await c.send(client, c.DEV_URL, key, body, f"hit-{i+1}", quiet=True)
                c.check(f"hit-{i+1} is HIT-EXACT", reply_hit.x_cache == "HIT-EXACT",
                         reply_hit.x_cache)
                hit_times.append(reply_hit.wall_ms)

            avg_hit = sum(hit_times) / len(hit_times)
            c.info(f"MISS wall time: {miss_ms:.1f}ms")
            c.info(f"HIT wall times: {[f'{t:.1f}' for t in hit_times]}")
            c.info(f"HIT average: {avg_hit:.1f}ms")
            c.info(f"Speedup: {miss_ms / avg_hit:.1f}x" if avg_hit > 0 else "N/A")

            c.check("cache hit is faster than miss", avg_hit < miss_ms,
                     f"hit_avg={avg_hit:.1f}ms, miss={miss_ms:.1f}ms")
            c.check("cache hit is at least 2x faster",
                     avg_hit < miss_ms / 2 if miss_ms > 0 else False,
                     f"speedup={miss_ms / avg_hit:.1f}x" if avg_hit > 0 else "N/A")
            c.section("Semantic cache performance")
            semantic_seed = c.make_payload("How does Redis determine whether a requested value is already stored in its cache?")
            semantic_miss = await c.send(
                client, c.DEV_URL, key, semantic_seed, "semantic-miss",
            )
            c.check("semantic seed is MISS", semantic_miss.x_cache == "MISS",
                     semantic_miss.x_cache)

            semantic_ctx = c.ctx_for(semantic_seed, key)
            await c.wait_for_key(r, semantic_ctx.semantic_key, timeout=10)

            semantic_variant = c.make_payload("How does Redis check if the requested data exists in the cache before retrieving it?")
            semantic_hit = await c.send(
                client, c.DEV_URL, key, semantic_variant, "semantic-hit",
            )
            c.check("paraphrase is HIT-SEMANTIC",
                     semantic_hit.x_cache == "HIT-SEMANTIC", semantic_hit.x_cache)
            c.check("semantic hit is faster than cache miss",
                     semantic_hit.x_cache == "HIT-SEMANTIC"
                     and semantic_hit.wall_ms < semantic_miss.wall_ms,
                     f"hit={semantic_hit.wall_ms:.1f}ms, "
                     f"miss={semantic_miss.wall_ms:.1f}ms")
            c.section("X-Exec-Time-Ms header accuracy")
            reply_check = await c.send(client, c.DEV_URL, key, body, "timing-check")
            if reply_check.exec_ms is not None:
                c.check("X-Exec-Time-Ms ≤ wall time",
                         reply_check.exec_ms <= reply_check.wall_ms + 10,  # +10ms grace
                         f"exec={reply_check.exec_ms:.1f}ms, wall={reply_check.wall_ms:.1f}ms")
                c.check("X-Exec-Time-Ms > 0", reply_check.exec_ms > 0,
                         f"exec={reply_check.exec_ms:.1f}ms")
            else:
                c.warn("X-Exec-Time-Ms header not present")
            c.section("Streaming cache performance")
            body_s = c.make_payload("What is the speed of light in km/s? Just the number.",
                                     stream=True)
            reply_stream_miss = await c.send(client, c.DEV_URL, key, body_s, "stream-miss")
            stream_miss_ms = reply_stream_miss.wall_ms

            ctx_s = c.ctx_for(body_s, key)
            await c.wait_for_key(r, ctx_s.exact_key, timeout=15)

            reply_stream_hit = await c.send(client, c.DEV_URL, key, body_s, "stream-hit")
            c.check("stream replay is HIT-EXACT", reply_stream_hit.x_cache == "HIT-EXACT",
                     reply_stream_hit.x_cache)
            c.check("stream hit is faster than miss",
                     reply_stream_hit.wall_ms < stream_miss_ms,
                     f"hit={reply_stream_hit.wall_ms:.1f}ms, miss={stream_miss_ms:.1f}ms")
            c.section("Performance summary")
            c.info(f"Non-streaming: MISS={miss_ms:.0f}ms, HIT avg={avg_hit:.0f}ms, "
                   f"speedup={miss_ms / avg_hit:.0f}x" if avg_hit > 0 else "N/A")
            c.info(f"Streaming:     MISS={stream_miss_ms:.0f}ms, "
                   f"HIT={reply_stream_hit.wall_ms:.0f}ms, "
                   f"speedup={stream_miss_ms / reply_stream_hit.wall_ms:.0f}x"
                   if reply_stream_hit.wall_ms > 0 else "N/A")
            c.section("Cleanup")
            await c.cleanup_tracked(r)
            await c.cleanup_scope(r, scope)

        finally:
            await r.aclose()


c.run(main)
