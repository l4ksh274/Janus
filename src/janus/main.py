"""Semantic caching proxy for the Responses and Chat Completions APIs.

The application validates a shared proxy key, checks Redis for exact or semantic
hits, and otherwise forwards the request to the configured upstream model
provider while keeping cache metadata for future reuse.
"""

import asyncio
import hashlib
import hmac
import json
import logging
import time
from collections.abc import AsyncIterator, Callable, Coroutine
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Annotated, Any

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from fastembed import TextEmbedding
from redis.asyncio import Redis

from janus import cache, embeddings
from janus.config import ProxyConfiguration, load_configuration
from janus.metrics import (
    instrument_app,
    record_cache_hit,
    record_request_latency,
    reset_metrics
)
from janus.models import ChatCompletionRequest, ResponseRequest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

UpstreamReply = tuple[int, bytes, str]
@dataclass(slots=True)
class Resources:
    """Shared resources used by request handlers during a process lifetime."""

    config: ProxyConfiguration
    http: httpx.AsyncClient
    redis: Redis
    embed_model: TextEmbedding
    embed_sem: asyncio.Semaphore
    semantic_enabled: bool
    inflight: dict[str, "asyncio.Future[UpstreamReply]"] = field(default_factory=dict)
    tasks: set["asyncio.Task[None]"] = field(default_factory=set)


def _load_model() -> TextEmbedding:
    """Load and warm the embedding model before the app starts serving traffic."""
    model = TextEmbedding(model_name=embeddings.EMBED_MODEL_NAME)
    list(model.embed(["warmup text"] * 4))
    return model


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the shared proxy resources once during FastAPI startup."""
    config = load_configuration()
    environment = config.environment
    app_settings = config.app
    async with (
        httpx.AsyncClient(
            base_url=environment.upstream_base_url,
            headers={"Authorization": f"Bearer {environment.upstream_api_key}"},
            timeout=httpx.Timeout(120.0, connect=10.0),
        ) as http,
        Redis.from_url(environment.redis_url) as redis,
    ):
        await redis.ping()
        semantic_enabled = await cache.ensure_index(redis)
        model = await asyncio.to_thread(_load_model)

        resources = Resources(
            config=config,
            http=http,
            redis=redis,
            embed_model=model,
            embed_sem=asyncio.Semaphore(app_settings.performance.max_concurrent_embeds),
            semantic_enabled=semantic_enabled,
        )
        app.state.resources = resources
        logger.info(
            "Ready (BASE_URL=%s, semantic_cache=%s)",
            environment.upstream_base_url,
            semantic_enabled,
        )
        yield

        if resources.tasks:
            await asyncio.gather(*resources.tasks, return_exceptions=True)


app = FastAPI(lifespan=lifespan)
instrument_app(app)


def get_resources(request: Request) -> Resources:
    """Return the process-wide resource bundle for this request."""
    return request.app.state.resources


ResourcesDep = Annotated[Resources, Depends(get_resources)]


@dataclass(frozen=True, slots=True)
class Client:
    """A validated client identity scoped to a cache namespace."""

    scope: str


async def authenticate(
    res: ResourcesDep,
    authorization: Annotated[str | None, Header()] = None,
) -> Client:
    """Validate the caller's bearer token and derive a per-client cache scope."""
    token = ""
    if authorization and authorization.startswith("Bearer "):
        token = authorization.removeprefix("Bearer ").strip()

    matches = [
        hmac.compare_digest(token.encode(), key.encode())
        for key in res.config.environment.proxy_key_values
    ]
    if not token or not any(matches):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return Client(scope=hashlib.sha256(token.encode()).hexdigest()[:16])


ClientDep = Annotated[Client, Depends(authenticate)]


def spawn(res: Resources, coro: Coroutine[Any, Any, None]) -> None:
    """Start a background task and keep its lifecycle tracked on the resources object."""
    task = asyncio.create_task(coro)
    res.tasks.add(task)
    task.add_done_callback(res.tasks.discard)


async def embed(res: Resources, text: str):
    """Embed a prompt in a worker thread under the process-wide concurrency limit."""
    async with res.embed_sem:
        return await asyncio.to_thread(embeddings.generate_vector, res.embed_model, text)


def is_completed_json(body: bytes) -> bool:
    """Return True only when a Responses payload represents a completed upstream reply."""
    try:
        data = json.loads(body)
    except ValueError:
        return False
    return isinstance(data, dict) and data.get("status") == "completed" and not data.get("error")


def is_completed_completion(body: bytes) -> bool:
    """Return True when a chat-completions payload is complete enough to cache."""
    try:
        data = json.loads(body)
    except ValueError:
        return False
    if not isinstance(data, dict) or data.get("error"):
        return False
    choices = data.get("choices", [])
    return bool(choices) and choices[0].get("finish_reason") == "stop"


def sse_error(message: str) -> bytes:
    """Build a minimal SSE error event for the client-facing stream."""
    return f"event: error\ndata: {json.dumps({'error': message})}\n\n".encode()


def upstream_error(status: int, public_message: str, exc: Exception) -> HTTPException:
    """Log an upstream failure and return a generic public error to the caller."""
    logger.error("%s (%s): %s", public_message, type(exc).__name__, exc)
    return HTTPException(status_code=status, detail=public_message)


def cached_reply(body: bytes, streaming: bool, kind: str, start: float) -> Response:
    """Return a cached response with the cache metadata headers attached."""
    exec_ms = (time.perf_counter() - start) * 1000
    return Response(
        content=body,
        status_code=200,
        media_type="text/event-stream" if streaming else "application/json",
        headers={"X-Cache": kind, "X-Exec-Time-Ms": f"{exec_ms:.2f}"},
    )


async def forward_json(
    res: Resources, payload: dict[str, Any], endpoint: str = "responses",
) -> UpstreamReply:
    """Send a JSON request to the configured upstream endpoint and return the raw reply."""
    try:
        upstream = await res.http.post(
            endpoint, json=payload, headers={"Accept": "application/json"}
        )
    except httpx.TimeoutException as exc:
        raise upstream_error(504, "Upstream timed out", exc) from exc
    except httpx.RequestError as exc:
        raise upstream_error(502, "Upstream unreachable", exc) from exc

    if upstream.status_code >= 400:
        logger.error("Upstream HTTP %s (%d bytes)", upstream.status_code, len(upstream.content))
    return (
        upstream.status_code,
        upstream.content,
        upstream.headers.get("content-type", "application/json"),
    )


async def forward_coalesced(
    res: Resources,
    ctx: cache.CacheCtx,
    payload: dict[str, Any],
    *,
    endpoint: str = "responses",
    completed_check: Callable[[bytes], bool] = is_completed_json,
) -> UpstreamReply:
    """Forward one upstream request and share it across concurrent identical calls."""
    existing = res.inflight.get(ctx.exact_key)
    if existing is not None:
        return await asyncio.shield(existing)

    fut: asyncio.Future[UpstreamReply] = asyncio.get_running_loop().create_future()
    res.inflight[ctx.exact_key] = fut
    try:
        reply = await forward_json(res, payload, endpoint)
    except HTTPException as exc:
        fut.set_exception(exc)
        fut.exception()
        raise
    except BaseException:
        fut.set_exception(HTTPException(status_code=503, detail="Request aborted"))
        fut.exception()
        raise
    else:
        fut.set_result(reply)
        status, body, _ = reply
        if status == 200 and completed_check(body):
            cache_settings = res.config.app.cache
            spawn(
                res,
                cache.store(
                    res.redis,
                    ctx,
                    body,
                    cache_settings.ttl_seconds,
                    exact_enabled=cache_settings.exact_enabled,
                    semantic_enabled=cache_settings.semantic_enabled and res.semantic_enabled,
                ),
            )
        return reply
    finally:
        res.inflight.pop(ctx.exact_key, None)


async def stream_and_close(
    upstream: httpx.Response,
    on_complete: Callable[[bytes], None] | None,
    max_buffer: int,
    completion_marker: bytes = b"event: response.completed",
) -> AsyncIterator[bytes]:
    """Relay an SSE stream and persist the buffered body when the upstream completes."""
    buffer: bytearray | None = bytearray() if on_complete else None
    clean = False
    try:
        async for chunk in upstream.aiter_bytes():
            if buffer is not None:
                if len(buffer) + len(chunk) <= max_buffer:
                    buffer.extend(chunk)
                else:
                    buffer = None
            yield chunk
        clean = True
    except (httpx.HTTPError, httpx.StreamError) as exc:
        logger.error("Upstream stream dropped: %s", exc)
        yield sse_error("Upstream connection lost")
    finally:
        await upstream.aclose()

    if clean and on_complete and buffer is not None and completion_marker in buffer:
        on_complete(bytes(buffer))


@app.post("/v1/responses")
async def forward_completion(
    body: ResponseRequest,
    res: ResourcesDep,
    client: ClientDep,
) -> Response:
    """Proxy a Responses API request through the cache and upstream provider."""
    start = time.perf_counter()
    payload: dict[str, Any] = body.model_dump(exclude_none=True)
    model_name = payload.get("model", "unknown")

    cache_settings = res.config.app.cache
    ctx: cache.CacheCtx | None = None
    cache_result = "BYPASS"

    cache_available = cache_settings.exact_enabled or (
        cache_settings.semantic_enabled and res.semantic_enabled
    )
    if cache_available and cache.is_cacheable(payload):
        ctx = cache.CacheCtx.build(payload, client.scope)
        cache_result = "MISS"

        if cache_settings.exact_enabled:
            hit = await cache.get_exact(res.redis, ctx.exact_key)
            if hit is not None:
                cache_result = "HIT-EXACT"
                record_cache_hit("exact", model_name, client.scope)
                elapsed = time.perf_counter() - start
                record_request_latency("POST", "/v1/responses", cache_result, elapsed)
                logger.info("L1 EXACT HIT in %.2f ms", elapsed * 1000)
                return cached_reply(hit, body.stream, cache_result, start)

        text = (
            embeddings.semantic_text(body.input)
            if cache_settings.semantic_enabled and res.semantic_enabled
            else None
        )
        if text is not None:
            embed_start = time.perf_counter()
            ctx.vector = await embed(res, text)
            ctx.prompt = text
            embed_ms = (time.perf_counter() - embed_start) * 1000
            logger.debug("embedded in %.2f ms shape=%s", embed_ms, ctx.vector.shape)

            hit = await cache.search_semantic(
                res.redis,
                ctx.scope,
                ctx.fp,
                ctx.vector,
                cache_settings.max_semantic_distance,
                prompt=text,
            )
            if hit is not None:
                cache_result = "HIT-SEMANTIC"
                record_cache_hit("semantic", model_name, client.scope)
                elapsed = time.perf_counter() - start
                record_request_latency("POST", "/v1/responses", cache_result, elapsed)
                logger.info("L2 SEMANTIC HIT in %.2f ms", elapsed * 1000)
                return cached_reply(hit, body.stream, cache_result, start)

    cache_header = {"X-Cache": "MISS" if ctx else "BYPASS"}

    if not body.stream:
        logger.info("Forwarding non-streaming POST to %sresponses", res.http.base_url)
        if ctx is not None:
            status, content, media_type = await forward_coalesced(res, ctx, payload)
        else:
            status, content, media_type = await forward_json(res, payload)
        elapsed = time.perf_counter() - start
        record_request_latency("POST", "/v1/responses", cache_result, elapsed)
        return Response(
            content=content, status_code=status, media_type=media_type, headers=cache_header
        )

    req = res.http.build_request(
        "POST", "responses", json=payload, headers={"Accept": "text/event-stream"}
    )
    logger.info("Forwarding streaming POST to %s", req.url)
    try:
        upstream = await res.http.send(req, stream=True)
    except httpx.TimeoutException as exc:
        raise upstream_error(504, "Upstream timed out", exc) from exc
    except httpx.RequestError as exc:
        raise upstream_error(502, "Upstream unreachable", exc) from exc

    if upstream.status_code >= 400:
        try:
            content = await upstream.aread()
        finally:
            await upstream.aclose()
        logger.error("Upstream HTTP %s (%d bytes)", upstream.status_code, len(content))
        elapsed = time.perf_counter() - start
        record_request_latency("POST", "/v1/responses", cache_result, elapsed)
        return Response(
            content=content,
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type", "application/json"),
        )

    on_complete: Callable[[bytes], None] | None = None
    if ctx is not None:
        stream_ctx = ctx

        def store_completed_stream(buffered: bytes) -> None:
            spawn(
                res,
                cache.store(
                    res.redis,
                    stream_ctx,
                    buffered,
                    cache_settings.ttl_seconds,
                    exact_enabled=cache_settings.exact_enabled,
                    semantic_enabled=cache_settings.semantic_enabled and res.semantic_enabled,
                ),
            )

        on_complete = store_completed_stream

    elapsed = time.perf_counter() - start
    record_request_latency("POST", "/v1/responses", cache_result, elapsed)

    return StreamingResponse(
        stream_and_close(
            upstream, on_complete, res.config.app.performance.max_stream_buffer_bytes
        ),
        media_type="text/event-stream",
        headers=cache_header,
    )



@app.post("/v1/chat/completions")
async def forward_chat_completion(

    body: ChatCompletionRequest,
    res: ResourcesDep,
    client: ClientDep,
) -> Response:
    start = time.perf_counter()
    payload: dict[str, Any] = body.model_dump(exclude_none=True)
    model_name = payload.get("model", "unknown")

    cache_settings = res.config.app.cache
    ctx: cache.CacheCtx | None = None
    cache_result = "BYPASS"

    cache_available = cache_settings.exact_enabled or (
        cache_settings.semantic_enabled and res.semantic_enabled
    )
    if cache_available and cache.is_cacheable_completion(payload):
        ctx = cache.CacheCtx.build(payload, client.scope)
        cache_result = "MISS"

        if cache_settings.exact_enabled:
            hit = await cache.get_exact(res.redis, ctx.exact_key)
            if hit is not None:
                cache_result = "HIT-EXACT"
                record_cache_hit("exact", model_name, client.scope)
                elapsed = time.perf_counter() - start
                record_request_latency("POST", "/v1/chat/completions", cache_result, elapsed)
                logger.info("L1 EXACT HIT in %.2f ms", elapsed * 1000)
                return cached_reply(hit, body.stream, cache_result, start)

        text = (
            embeddings.semantic_text_from_messages(body.messages)
            if cache_settings.semantic_enabled and res.semantic_enabled else None
        )
        if text is not None:
            embed_start = time.perf_counter()
            ctx.vector = await embed(res, text)
            ctx.prompt = text
            embed_ms = (time.perf_counter() - embed_start) * 1000
            logger.debug("embedded in %.2f ms shape=%s", embed_ms, ctx.vector.shape)

            hit = await cache.search_semantic(
                res.redis,
                ctx.scope,
                ctx.fp,
                ctx.vector,
                cache_settings.max_semantic_distance,
                prompt=text,
            )
            if hit is not None:
                cache_result = "HIT-SEMANTIC"
                record_cache_hit("semantic", model_name, client.scope)
                elapsed = time.perf_counter() - start
                record_request_latency("POST", "/v1/chat/completions", cache_result, elapsed)
                logger.info("L2 SEMANTIC HIT in %.2f ms", elapsed * 1000)
                return cached_reply(hit, body.stream, cache_result, start)

    cache_header = {"X-Cache": "MISS" if ctx else "BYPASS"}

    if not body.stream:
        logger.info("Forwarding non-streaming POST to %schat/completions", res.http.base_url)
        if ctx is not None:
            status, content, media_type = await forward_coalesced(
                res, ctx, payload,
                endpoint="chat/completions",
                completed_check=is_completed_completion,
            )
        else:
            status, content, media_type = await forward_json(res, payload, "chat/completions")
        elapsed = time.perf_counter() - start
        record_request_latency("POST", "/v1/chat/completions", cache_result, elapsed)
        return Response(
            content=content, status_code=status, media_type=media_type, headers=cache_header
        )

    req = res.http.build_request(
        "POST", "chat/completions", json=payload, headers={"Accept": "text/event-stream"}
    )
    logger.info("Forwarding streaming POST to %s", req.url)
    try:
        upstream = await res.http.send(req, stream=True)
    except httpx.TimeoutException as exc:
        raise upstream_error(504, "Upstream timed out", exc) from exc
    except httpx.RequestError as exc:
        raise upstream_error(502, "Upstream unreachable", exc) from exc

    if upstream.status_code >= 400:
        try:
            content = await upstream.aread()
        finally:
            await upstream.aclose()
        logger.error("Upstream HTTP %s (%d bytes)", upstream.status_code, len(content))
        elapsed = time.perf_counter() - start
        record_request_latency("POST", "/v1/chat/completions", cache_result, elapsed)
        return Response(
            content=content,
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type", "application/json"),
        )

    on_complete: Callable[[bytes], None] | None = None
    if ctx is not None:
        stream_ctx = ctx

        def store_completed_stream(buffered: bytes) -> None:
            spawn(
                res,
                cache.store(
                    res.redis,
                    stream_ctx,
                    buffered,
                    cache_settings.ttl_seconds,
                    exact_enabled=cache_settings.exact_enabled,
                    semantic_enabled=cache_settings.semantic_enabled and res.semantic_enabled,
                ),
            )

        on_complete = store_completed_stream

    elapsed = time.perf_counter() - start
    record_request_latency("POST", "/v1/chat/completions", cache_result, elapsed)

    return StreamingResponse(
        stream_and_close(
            upstream, on_complete, res.config.app.performance.max_stream_buffer_bytes,
            completion_marker=b"data: [DONE]",
        ),
        media_type="text/event-stream",
        headers=cache_header,
    )


@app.post("/reset/metrics")
async def reset_metrics_endpoint():
    reset_metrics()
    return {"status": "success", "message": "Metrics reset"}