import logging
import sys
from contextlib import asynccontextmanager
from os import getenv
from typing import Annotated, Any

import httpx
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from httpx import AsyncClient, HTTPError, Timeout, StreamError
from redis.asyncio import Redis

from janus.models import ResponseRequest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

API_KEY = getenv("NVIDIA_API_KEY", "")
BASE_URL = getenv("BASE_URL", "").rstrip("/") + "/"
REDIS_URL = getenv("REDIS_URL", "redis://localhost:6379")

if not API_KEY or BASE_URL == "/":
    sys.exit("NVIDIA_API_KEY and BASE_URL must be set")


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with (
        AsyncClient(
            base_url=BASE_URL,
            headers={"Authorization": f"Bearer {API_KEY}"},
            timeout=Timeout(120.0, connect=10.0),
        ) as client,
        Redis.from_url(REDIS_URL, decode_responses=True) as redis,
    ):
        await redis.ping()
        logger.info("HTTP client and Redis ready (BASE_URL=%s)", BASE_URL)

        app.state.http = client
        app.state.redis = redis
        yield


app = FastAPI(lifespan=lifespan)


def get_http(request: Request) -> AsyncClient:
    return request.app.state.http


def get_redis(request: Request) -> Redis:
    return request.app.state.redis


HttpClient = Annotated[AsyncClient, Depends(get_http)]
RedisClient = Annotated[Redis, Depends(get_redis)]


async def stream_and_close(response: httpx.Response):
    try:
        async for chunk in response.aiter_bytes():
            yield chunk
    except (HTTPError, StreamError) as exc:
        logger.error(f"Upstream stream dropped: {exc}")
        yield b"event: error\ndata: {\"error\": \"Upstream connection lost\"}\n\n"
    finally:
        await response.aclose()


@app.post("/v1/responses")
async def forward_completion(body: ResponseRequest, http: HttpClient):
    payload: dict[str, Any] = body.model_dump(exclude_none=True)

    try:
        if not body.stream:
            logger.info("Forwarding non-streaming POST to %sresponses", http.base_url)
            upstream = await http.post(
                "responses", json=payload, headers={"Accept": "application/json"}
            )
            return Response(
                content=upstream.content,
                status_code=upstream.status_code,
                media_type=upstream.headers.get("content-type", "application/json"),
            )

        req = http.build_request(
            "POST",
            "responses",
            json=payload,
            headers={"Accept": "text/event-stream"},
        )
        logger.info("Forwarding streaming POST to %s", req.url)
        upstream = await http.send(req, stream=True)
    except httpx.TimeoutException as exc:
        logger.error("Upstream timed out (%s): %s", type(exc).__name__, exc)
        raise HTTPException(
            status_code=504,
            detail=f"Upstream timed out ({type(exc).__name__}): {exc}",
        )
    except httpx.RequestError as exc:
        logger.error("Upstream request failed (%s): %s", type(exc).__name__, exc)
        raise HTTPException(
            status_code=502,
            detail=f"Upstream unreachable ({type(exc).__name__}): {exc}",
        )

    if upstream.status_code >= 400:
        try:
            content = await upstream.aread()
        finally:
            await upstream.aclose()

        logger.error("Upstream HTTP %s: %s", upstream.status_code, content[:500])
        return Response(
            content=content,
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type", "application/json"),
        )

    return StreamingResponse(
        stream_and_close(upstream),
        media_type="text/event-stream",
    )