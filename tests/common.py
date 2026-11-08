"""Shared helpers for the Janus semantic-cache test scripts.

Every script imports this module as ``common`` to log test output, issue HTTP
requests against the proxy, and inspect Redis state and cache metadata.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import statistics
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable

import httpx

TESTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEV_URL = os.getenv("DEV_URL", "http://localhost:8000").rstrip("/")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
TEST_MODEL = os.getenv("TEST_MODEL", "meta/llama-3.1-8b-instruct")
APP_IMPORT = os.getenv("APP_IMPORT", "janus.main:app")
LOG_PACKAGE = os.getenv("LOG_PACKAGE", "janus")
MAX_DISTANCE = float(os.getenv("SEMANTIC_MAX_DISTANCE", "0.08"))
CACHE_TTL = int(os.getenv("CACHE_TTL_SECONDS", "3600"))
KEEP_KEYS = os.getenv("KEEP_KEYS") == "1"
MOCK_PORT = int(os.getenv("MOCK_PORT", "9099"))
PROXY_PORT = int(os.getenv("TEST_PROXY_PORT", "8099"))

NONCE = uuid.uuid4().hex[:8]

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_USE_COLOR = bool(os.getenv("FORCE_COLOR")) or (sys.stdout.isatty() and not os.getenv("NO_COLOR"))


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text


def proxy_keys() -> list[str]:
    """Return the valid proxy API keys configured for the current test run."""
    raw = os.getenv("PROXY_API_KEYS", "")
    return [k.strip() for k in raw.split(",") if k.strip()]


def mask(secret: str) -> str:
    """Shorten a bearer token for log output without revealing the whole secret."""
    if len(secret) <= 8:
        return "*" * len(secret)
    return f"{secret[:4]}...{secret[-2:]} (len={len(secret)})"


def short(text: Any, n: int = 90) -> str:
    """Truncate a value for readable console output while preserving the label."""
    s = str(text).replace("\n", "\\n")
    return s if len(s) <= n else s[: n - 3] + "..."


class _Tee:
    def __init__(self, stream, fh):
        self._stream, self._fh = stream, fh

    def write(self, s: str) -> int:
        self._stream.write(s)
        self._fh.write(ANSI_RE.sub("", s))
        return len(s)

    def flush(self) -> None:
        self._stream.flush()
        self._fh.flush()

    def isatty(self) -> bool:
        return self._stream.isatty()

    def __getattr__(self, name):
        return getattr(self._stream, name)


_results: list[tuple[str, str, str]] = []  # (status, name, detail)
_log_lines_from_package: list[str] = []
_script_name = "script"
_log_path: Path | None = None


def ts() -> str:
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def out(msg: str = "", indent: int = 0) -> None:
    pad = "    " * indent
    for line in str(msg).splitlines() or [""]:
        print(f"{_c('90', ts())} {pad}{line}", flush=True)


class _PackageLogHandler:
    """Prints log records emitted by the janus package while a script runs in-process."""

    def __init__(self):
        import logging

        class H(logging.Handler):
            def emit(inner, record):  # noqa: N805
                msg = record.getMessage()
                tail = "  (+traceback)" if record.exc_info else ""
                line = f"[{record.name} {record.levelname}] {msg}{tail}"
                _log_lines_from_package.append(line)
                colour = {"WARNING": "33", "ERROR": "31", "DEBUG": "90"}.get(record.levelname, "36")
                out(_c(colour, "  log> " + line))

        self.handler = H(level=logging.DEBUG)


def start(title: str, what: str) -> None:
    """Call first in every script: sets up the tee file, banner and package logging."""
    global _script_name, _log_path
    import logging

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    _script_name = Path(sys.argv[0]).stem
    log_dir = TESTS_DIR / "logs"
    log_dir.mkdir(exist_ok=True)
    _log_path = log_dir / f"{_script_name}-{datetime.now():%Y%m%d-%H%M%S}.log"
    sys.stdout = _Tee(sys.stdout, open(_log_path, "w", encoding="utf-8"))  # type: ignore[assignment]

    pkg_logger = logging.getLogger(LOG_PACKAGE)
    pkg_logger.setLevel(logging.DEBUG)
    pkg_logger.propagate = False
    pkg_logger.addHandler(_PackageLogHandler().handler)

    bar = "=" * 78
    print(_c("1;36", bar))
    print(_c("1;36", f"  {title}"))
    print(_c("36", f"  {what}"))
    print(_c("90", f"  run id: {NONCE}   log file: {_log_path.relative_to(PROJECT_ROOT)}"))
    print(_c("1;36", bar), flush=True)


def section(title: str) -> None:
    print()
    print(_c("1;34", f"--- {title} " + "-" * max(0, 70 - len(title))), flush=True)


def step(msg: str) -> None:
    out(_c("1", ">> " + msg))


def info(msg: str, indent: int = 1) -> None:
    out(_c("37", msg), indent)


def detail(label: str, value: Any, indent: int = 1) -> None:
    out(f"{_c('90', label + ':'):<30} {value}", indent)


def show(label: str, obj: Any, limit: int = 600, indent: int = 1) -> None:
    text = json.dumps(obj, indent=2, ensure_ascii=False, default=str) if not isinstance(obj, str) else obj
    if len(text) > limit:
        text = text[:limit] + f"\n... ({len(text) - limit} more chars)"
    out(_c("90", label + ":"), indent)
    out(text, indent + 1)


def warn(msg: str) -> None:
    _results.append(("WARN", msg, ""))
    out(_c("33", f"[WARN] {msg}"), 1)


def skip(msg: str) -> None:
    _results.append(("SKIP", msg, ""))
    out(_c("35", f"[SKIP] {msg}"), 1)


def watch_server(hint: str) -> None:
    """Tell the user which line to look for in THEIR dev-server terminal."""
    out(_c("1;35", f"  >> watch your dev-server terminal for: {hint}"), 1)


def check(name: str, ok: bool, evidence: Any = "") -> bool:
    status = "PASS" if ok else "FAIL"
    _results.append((status, name, str(evidence)))
    colour = "1;32" if ok else "1;31"
    line = f"[{status}] {name}"
    if evidence != "":
        line += f"  ->  {short(evidence, 200)}"
    out(_c(colour, line), 1)
    return ok


def finish() -> None:
    passed = sum(1 for s, *_ in _results if s == "PASS")
    failed = [r for r in _results if r[0] == "FAIL"]
    warns = [r for r in _results if r[0] == "WARN"]
    skips = [r for r in _results if r[0] == "SKIP"]
    print()
    print(_c("1;36", "=" * 78))
    print(_c("1", f"  SUMMARY ({_script_name}):  {passed} passed, {len(failed)} failed, "
                   f"{len(warns)} warnings, {len(skips)} skipped"))
    for _, name, ev in failed:
        print(_c("1;31", f"    FAILED: {name}  {short(ev, 160)}"))
    for _, name, _ in warns:
        print(_c("33", f"    WARN:   {name}"))
    print(_c("1;36", "=" * 78))
    if _log_path:
        print(_c("90", f"  full transcript saved to {_log_path}"))
    sys.stdout.flush()
    sys.exit(1 if failed else 0)


def run(coro_fn: Callable[[], Awaitable[None]]) -> None:
    """Run the script's async main(), always printing the summary."""
    try:
        asyncio.run(coro_fn())
    except KeyboardInterrupt:
        out(_c("31", "interrupted"))
        _results.append(("FAIL", "script interrupted", ""))
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 - we want the traceback in the log, then a summary
        import traceback

        out(_c("1;31", f"UNEXPECTED ERROR: {type(exc).__name__}: {exc}"))
        for line in traceback.format_exc().splitlines():
            out(_c("31", line), 1)
        _results.append(("FAIL", f"unexpected error: {type(exc).__name__}: {exc}", ""))
    finish()


# --------------------------------------------------------------------------- #
# Request helpers
# --------------------------------------------------------------------------- #
def make_payload(text: Any, *, instructions: str | None = None, **extra: Any) -> dict[str, Any]:
    """A cacheable (temperature 0, text-only, stateless) Responses API request."""
    body: dict[str, Any] = {
        "model": TEST_MODEL,
        "input": text,
        "temperature": 0,
        "instructions": instructions if instructions is not None else f"Be terse. [test-run {NONCE}]",
    }
    body.update(extra)
    return body


@dataclass
class Reply:
    status: int
    headers: httpx.Headers
    body: bytes
    wall_ms: float
    chunks: list[tuple[float, bytes]] = field(default_factory=list)  # (ms since send, bytes)
    ttfb_ms: float | None = None

    @property
    def x_cache(self) -> str | None:
        return self.headers.get("x-cache")

    @property
    def exec_ms(self) -> float | None:
        v = self.headers.get("x-exec-time-ms")
        return float(v) if v is not None else None

    @property
    def content_type(self) -> str:
        return self.headers.get("content-type", "")

    def json(self) -> Any:
        return json.loads(self.body)

    @property
    def events(self) -> list[tuple[str, Any]]:
        """Parse an SSE body into [(event_name, parsed_data_or_raw)]."""
        events: list[tuple[str, Any]] = []
        for block in self.body.decode("utf-8", "replace").split("\n\n"):
            name, data = "message", None
            for line in block.splitlines():
                if line.startswith("event:"):
                    name = line[6:].strip()
                elif line.startswith("data:"):
                    raw = line[5:].strip()
                    try:
                        data = json.loads(raw)
                    except ValueError:
                        data = raw
            if data is not None or name != "message":
                events.append((name, data))
        return events

    def answer(self) -> str:
        """Best-effort extraction of the model's text from a JSON or SSE body."""
        try:
            if "event-stream" in self.content_type:
                deltas = [d.get("delta", "") for n, d in self.events
                          if n == "response.output_text.delta" and isinstance(d, dict)]
                return "".join(deltas)
            data = self.json()
            texts: list[str] = []
            for item in data.get("output", []):
                for part in item.get("content", []) or []:
                    if isinstance(part, dict) and "text" in part:
                        texts.append(part["text"])
            return "".join(texts)
        except Exception:  # noqa: BLE001
            return self.body[:120].decode("utf-8", "replace")


def auth_headers(key: str | None) -> dict[str, str]:
    h = {"Content-Type": "application/json"}
    if key is not None:
        h["Authorization"] = f"Bearer {key}"
    return h


async def send(
    client: httpx.AsyncClient,
    base: str,
    key: str | None,
    body: dict[str, Any],
    label: str,
    *,
    raw: bytes | None = None,
    headers: dict[str, str] | None = None,
    track: bool = True,
    quiet: bool = False,
) -> Reply:
    """POST /v1/responses, print request + response summaries, return a Reply."""
    if track and key is not None:
        track_payload(body, key)
    data = raw if raw is not None else json.dumps(body).encode()
    h = auth_headers(key)
    h.update(headers or {})

    if not quiet:
        txt = body.get("input")
        if isinstance(txt, list):
            txt = f"<{len(txt)} message(s)> " + short(json.dumps(txt, ensure_ascii=False), 70)
        out(_c("1", f"-> [{label}] POST {base}/v1/responses  key={mask(key) if key else '<none>'}"), 1)
        out(f"stream={body.get('stream', False)} temperature={body.get('temperature', '<unset>')} "
            f"input={short(txt, 70)!r}", 2)

    t0 = time.perf_counter()
    chunks: list[tuple[float, bytes]] = []
    ttfb: float | None = None
    async with client.stream("POST", f"{base}/v1/responses", content=data, headers=h) as resp:
        async for chunk in resp.aiter_bytes():
            now = (time.perf_counter() - t0) * 1000
            if ttfb is None:
                ttfb = now
            chunks.append((now, chunk))
        status, hdrs = resp.status_code, resp.headers
    wall = (time.perf_counter() - t0) * 1000
    reply = Reply(status, hdrs, b"".join(c for _, c in chunks), wall, chunks, ttfb)

    if not quiet:
        parts = [f"status={reply.status}", f"X-Cache={reply.x_cache}"]
        if reply.exec_ms is not None:
            parts.append(f"X-Exec-Time-Ms={reply.exec_ms:.2f}")
        parts += [f"wall={wall:.1f}ms", f"bytes={len(reply.body)}", f"chunks={len(chunks)}"]
        out(_c("1", "<- " + "  ".join(parts)), 1)
        out(f"content-type={reply.content_type}", 2)
        if reply.status == 200:
            out(f"answer={short(reply.answer(), 110)!r}", 2)
        else:
            out(f"body={short(reply.body.decode('utf-8', 'replace'), 160)}", 2)
    return reply


# --------------------------------------------------------------------------- #
# Redis helpers
# --------------------------------------------------------------------------- #
def redis_client(url: str | None = None):
    from redis.asyncio import Redis

    return Redis.from_url(url or REDIS_URL)  # bytes client, same as the proxy


def scope_for(api_key: str) -> str:
    """Mirror of main.authenticate(): the per-client cache namespace."""
    return hashlib.sha256(api_key.encode()).hexdigest()[:16]


def proxy_view(body: dict[str, Any]) -> dict[str, Any]:
    """The payload exactly as the handler builds it (after ResponseRequest validation)."""
    from janus.models import ResponseRequest

    return ResponseRequest(**body).model_dump(exclude_none=True)


def ctx_for(body: dict[str, Any], api_key: str):
    from janus import cache

    return cache.CacheCtx.build(proxy_view(body), scope_for(api_key))


_tracked: set[str] = set()


def track_payload(body: dict[str, Any], api_key: str) -> None:
    try:
        ctx = ctx_for(body, api_key)
        _tracked.update({ctx.exact_key, ctx.semantic_key})
    except Exception:  # noqa: BLE001 - invalid bodies are fine, they just aren't tracked
        pass


async def cleanup_tracked(r) -> None:
    if KEEP_KEYS:
        out(f"KEEP_KEYS=1 -> leaving {len(_tracked)} test keys in Redis for manual inspection", 1)
        for k in sorted(_tracked):
            out(k, 2)
        return
    if _tracked:
        n = await r.delete(*_tracked)
        out(f"cleanup: deleted {n} test key(s) created by this run", 1)


async def cleanup_scope(r, scope: str) -> int:
    n = 0
    for pattern in (f"cache:exact:{scope}:*", f"cache:sem:{scope}:*"):
        keys = [k async for k in r.scan_iter(match=pattern, count=500)]
        if keys:
            n += await r.delete(*keys)
    return n


async def scope_keys(r, scope: str) -> list[str]:
    keys: list[str] = []
    for pattern in (f"cache:exact:{scope}:*", f"cache:sem:{scope}:*"):
        keys += [k.decode() async for k in r.scan_iter(match=pattern, count=500)]
    return sorted(keys)


async def describe_key(r, key: str, indent: int = 2) -> dict[str, Any]:
    """Print what Redis holds under `key`; return a dict of facts for assertions."""
    typ = (await r.type(key)).decode()
    facts: dict[str, Any] = {"type": typ, "exists": typ != "none"}
    if typ == "none":
        out(f"{key}  -> (does not exist)", indent)
        return facts
    ttl = await r.ttl(key)
    facts["ttl"] = ttl
    if typ == "string":
        val = await r.get(key)
        facts["value"] = val
        out(f"{short(key, 70)}  type=string ttl={ttl}s size={len(val)}B", indent)
    elif typ == "hash":
        h = await r.hgetall(key)
        facts["hash"] = h
        vec = h.get(b"vec", b"")
        out(f"{short(key, 70)}  type=hash ttl={ttl}s", indent)
        out(f"fields={sorted(k.decode() for k in h)}  vec={len(vec)}B "
            f"({len(vec) // 4} float32)  body={len(h.get(b'body', b''))}B", indent + 1)
    return facts


async def wait_for_key(r, key: str, timeout: float = 10.0) -> float | None:
    """Poll until `key` exists (background cache writes are asynchronous). Returns seconds waited."""
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        if await r.exists(key):
            return time.perf_counter() - t0
        await asyncio.sleep(0.05)
    return None


async def eventually(fn: Callable[[], Awaitable[Any]], timeout: float = 3.0, interval: float = 0.1) -> Any:
    """Poll an async callable until it returns a truthy value (or timeout)."""
    t0 = time.perf_counter()
    result = await fn()
    while not result and time.perf_counter() - t0 < timeout:
        await asyncio.sleep(interval)
        result = await fn()
    return result


def median(values: list[float]) -> float:
    return statistics.median(values) if values else float("nan")
