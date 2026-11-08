"""Start a proxy and mock upstream for end-to-end failure-path tests.

Each stack gets its own API keys and cache scope so the Redis data for one run
stays separate from another. The spawned proxy streams its log output into the
parent test session so request/response failures can be correlated with real
server-side logs.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

import httpx

import common as c
from mock_upstream import MockUpstream

UPSTREAM_SECRET = "mock-upstream-secret"


class ProxyProcess:
    """Manage the background proxy subprocess used during a test stack."""

    def __init__(self, name: str, env: dict[str, str], port: int = c.PROXY_PORT):
        self.name, self.port = name, port
        self.env = {**os.environ, **env, "TEST_PROXY_PORT": str(port),
                    "APP_IMPORT": c.APP_IMPORT, "LOG_PACKAGE": c.LOG_PACKAGE,
                    "PYTHONUNBUFFERED": "1",
                    "PYTHONPATH": os.pathsep.join([str(c.PROJECT_ROOT), os.environ.get("PYTHONPATH", "")])}
        self.lines: list[str] = []
        self._ready = threading.Event()
        self._proc: subprocess.Popen[str] | None = None
        self._reader: threading.Thread | None = None

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def _read(self) -> None:
        assert self._proc and self._proc.stdout
        for line in self._proc.stdout:
            line = line.rstrip("\n")
            self.lines.append(line)
            if "Ready (" in line:
                self._ready.set()
            colour = "31" if "ERROR" in line or "Traceback" in line else "33" if "WARNING" in line else "90"
            c.out(c._c(colour, f"  proxy| {line}"))

    def spawn(self) -> None:
        c.out(f"starting proxy '{self.name}' on {self.base}  (app={c.APP_IMPORT})", 1)
        self._proc = subprocess.Popen(
            [sys.executable, "-u", str(c.TESTS_DIR / "_run_proxy.py")],
            env=self.env, cwd=str(c.PROJECT_ROOT),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()

    async def start(self, timeout: float = 240.0) -> None:
        self.spawn()
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < timeout:
            if self._ready.is_set():
                c.out(f"proxy '{self.name}' is ready after {time.perf_counter() - t0:.1f}s "
                      f"(includes embedding-model load)", 1)
                return
            if self._proc and self._proc.poll() is not None:
                raise RuntimeError(f"proxy exited during startup with code {self._proc.returncode}")
            await asyncio.sleep(0.1)
        raise RuntimeError("proxy did not become ready in time")

    async def wait_exit(self, timeout: float = 120.0) -> int | None:
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < timeout:
            if self._proc and self._proc.poll() is not None:
                await asyncio.sleep(0.3)  # let the reader drain
                return self._proc.returncode
            await asyncio.sleep(0.1)
        return None

    def stop(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self._proc.kill()

    def mark(self) -> int:
        return len(self.lines)

    async def expect_log(self, needle: str, since: int = 0, timeout: float = 3.0) -> str | None:
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < timeout:
            for line in self.lines[since:]:
                if needle in line:
                    return line
            await asyncio.sleep(0.05)
        return None

    def logs_since(self, since: int) -> list[str]:
        return self.lines[since:]


@dataclass
class Stack:
    """Bundle the mock upstream, proxy, Redis client, and auth keys used by a test."""

    mock: MockUpstream
    proxy: ProxyProcess
    redis: Any
    key_a: str
    key_b: str
    client: httpx.AsyncClient

    @property
    def base(self) -> str:
        return self.proxy.base

    @property
    def scope_a(self) -> str:
        return c.scope_for(self.key_a)

    @property
    def scope_b(self) -> str:
        return c.scope_for(self.key_b)

    async def send(self, body: dict[str, Any], label: str, key: str | None = None, **kw: Any) -> c.Reply:
        return await c.send(self.client, self.base, key or self.key_a, body, label, track=False, **kw)


@contextlib.asynccontextmanager
async def stack(
    name: str,
    *,
    extra_env: dict[str, str] | None = None,
    use_mock: bool = True,
    dead_upstream: bool = False,
) -> AsyncIterator[Stack]:
    """Mock upstream + proxy + Redis client, with scope cleanup on exit."""
    c.section(f"Setting up isolated test stack: {name}")
    mock = MockUpstream()
    if use_mock:
        await mock.start()
    key_a, key_b = f"test-key-A-{c.NONCE}", f"test-key-B-{c.NONCE}"
    base_url = "http://127.0.0.1:1/v1" if dead_upstream else mock.url
    env = {
        "NVIDIA_API_KEY": UPSTREAM_SECRET,
        "BASE_URL": base_url,
        "PROXY_API_KEYS": f"{key_a},{key_b}",
        "REDIS_URL": c.REDIS_URL,
        **(extra_env or {}),
    }
    proxy = ProxyProcess(name, env)
    redis = c.redis_client()
    client = httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=5.0))
    try:
        await proxy.start()
        yield Stack(mock, proxy, redis, key_a, key_b, client)
    finally:
        c.section(f"Tearing down test stack: {name}")
        await client.aclose()
        proxy.stop()
        await mock.stop()
        try:
            for scope in (c.scope_for(key_a), c.scope_for(key_b)):
                n = await c.cleanup_scope(redis, scope)
                c.out(f"cleanup: removed {n} key(s) from scope {scope}", 1)
        finally:
            await redis.aclose()


def mock_body(text: str, mode: str = "ok", **extra: Any) -> dict[str, Any]:
    """Create a cacheable request whose upstream behavior is selected via `metadata.mock`."""
    meta = {"mock": mode}
    meta.update({k: v for k, v in extra.items() if k in ("delay",)})
    rest = {k: v for k, v in extra.items() if k not in ("delay",)}
    return c.make_payload(text, instructions="mock test", metadata=meta, model="mock-model", **rest)
