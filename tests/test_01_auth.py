#!/usr/bin/env python3
"""Exercises the proxy authentication path.

This script verifies that valid bearer tokens succeed and that missing,
invalid, or malformed credentials produce 401 responses.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
os.environ.setdefault("TEST_MODEL", "nvidia/nemotron-3-ultra-550b-a55b")

import common as c  # noqa: E402
import httpx  # noqa: E402

TITLE = "01 — Authentication & Connectivity"


async def main() -> None:
    c.start(TITLE, "Verify proxy auth accepts/rejects requests correctly")
    keys = c.proxy_keys()
    if not keys:
        c.check("PROXY_API_KEYS configured", False, "env var is empty")
        return
    key = keys[0]
    c.info(f"Using key: {c.mask(key)}")
    c.info(f"Proxy URL: {c.DEV_URL}")

    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=5.0)) as client:
        c.section("Valid API key")
        body = c.make_payload("Say only: hello")
        reply = await c.send(client, c.DEV_URL, key, body, "valid-key")
        c.check("valid key → status 200", reply.status == 200, f"status={reply.status}")
        c.check("X-Cache header present", reply.x_cache is not None, reply.x_cache)
        c.section("Missing API key")
        reply_no_key = await c.send(client, c.DEV_URL, None, body, "no-key")
        c.check("no key → status 401", reply_no_key.status == 401, f"status={reply_no_key.status}")
        c.section("Invalid API key")
        reply_bad = await c.send(client, c.DEV_URL, "totally-wrong-key-12345", body, "bad-key")
        c.check("bad key → status 401", reply_bad.status == 401, f"status={reply_bad.status}")
        c.section("Malformed auth header (no 'Bearer ' prefix)")
        h = {"Content-Type": "application/json", "Authorization": key}  # no "Bearer "
        raw = json.dumps(body).encode()
        t0 = __import__("time").perf_counter()
        async with client.stream("POST", f"{c.DEV_URL}/v1/responses", content=raw, headers=h) as resp:
            raw_body = b"".join([chunk async for chunk in resp.aiter_bytes()])
            status = resp.status_code
            hdrs = resp.headers
        wall = (__import__("time").perf_counter() - t0) * 1000
        malformed_reply = c.Reply(status, hdrs, raw_body, wall)
        c.check("malformed auth → 401", malformed_reply.status == 401, f"status={malformed_reply.status}")
        c.section("401 response body validation")
        try:
            err_body = reply_no_key.json()
            c.check("401 body has 'detail' field", "detail" in err_body, err_body)
            c.check("detail says 'Invalid or missing API key'",
                     err_body.get("detail") == "Invalid or missing API key",
                     err_body.get("detail"))
        except Exception as exc:
            c.check("401 body is JSON", False, str(exc))
        c.section("Cleanup")
        r = c.redis_client()
        try:
            await c.cleanup_tracked(r)
        finally:
            await r.aclose()


c.run(main)
