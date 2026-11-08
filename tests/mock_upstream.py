"""A controllable fake of the upstream Responses API (the thing BASE_URL points at).

Why: the proxy's caching, coalescing and failure handling can only be proven if we
know exactly how many times upstream was called and can make it misbehave on demand.

Behaviour is chosen per request with ``metadata.mock`` (metadata is forwarded by the
proxy but excluded from the cache key, so always vary ``input`` between cases):

    ok            200 JSON, status=completed       (default)
    failed        200 JSON, status=failed
    incomplete    200 JSON, status=incomplete
    error_field   200 JSON, status=completed but with an "error" object
    notjson       200 text/plain
    http500       500 JSON error whose body contains a secret-looking string
    http429       429 JSON error

    streaming (stream=true):
    ok            created, N deltas, response.completed
    no_completed  created, N deltas, then the stream just ends
    drop          created, a delta, then the TCP connection is cut
    big           lots of large deltas then completed (to exceed the buffer limit)
    http500       500 before any streaming starts

``metadata.delay`` (seconds) delays the reply. Every answer embeds a per-process call
number (``MOCK-ANSWER#3``) so a cached replay is recognisable: the number does not change.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

import common as c


@dataclass
class Call:
    n: int
    payload: dict[str, Any]
    authorization: str | None
    accept: str | None
    mode: str


@dataclass
class MockUpstream:
    port: int = c.MOCK_PORT
    calls: list[Call] = field(default_factory=list)
    _server: uvicorn.Server | None = None
    _task: asyncio.Task | None = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    @property
    def count(self) -> int:
        return len(self.calls)

    def reset(self) -> None:
        self.calls.clear()

    # ------------------------------------------------------------------ #
    def _build_app(self) -> FastAPI:
        app = FastAPI()

        def text_of(payload: dict[str, Any]) -> str:
            inp = payload.get("input")
            if isinstance(inp, str):
                return inp
            return json.dumps(inp, ensure_ascii=False)[:80]

        @app.post("/v1/responses")
        async def responses(request: Request):
            payload = await request.json()
            meta = payload.get("metadata") or {}
            mode = str(meta.get("mock", "ok"))
            delay = float(meta.get("delay", 0))
            call = Call(len(self.calls) + 1, payload, request.headers.get("authorization"),
                        request.headers.get("accept"), mode)
            self.calls.append(call)
            c.out(c._c("90", f"  mock-upstream> call #{call.n} stream={bool(payload.get('stream'))} "
                              f"mode={mode} delay={delay}s accept={call.accept} "
                              f"auth={'Bearer ***' if call.authorization else None}"))
            if delay:
                await asyncio.sleep(delay)

            answer = f"MOCK-ANSWER#{call.n} for: {text_of(payload)[:60]}"

            if payload.get("stream"):
                return self._stream(call, answer, payload)

            if mode == "http500":
                return JSONResponse({"error": {"message": "boom at SECRET-INTERNAL-HOST.corp:5432"}}, 500)
            if mode == "http429":
                return JSONResponse({"error": {"message": "rate limited"}}, 429)
            if mode == "notjson":
                return PlainTextResponse("hello, I am not JSON")

            body = {
                "id": f"resp_mock_{call.n}",
                "object": "response",
                "model": payload.get("model"),
                "status": {"failed": "failed", "incomplete": "incomplete"}.get(mode, "completed"),
                "output": [{"type": "message", "role": "assistant",
                            "content": [{"type": "output_text", "text": answer}]}],
            }
            if mode == "error_field":
                body["error"] = {"message": "something went wrong"}
            return JSONResponse(body)

        return app

    def _stream(self, call: Call, answer: str, payload: dict[str, Any]):
        mode = call.mode
        if mode == "http500":
            return JSONResponse({"error": {"message": "stream boom SECRET-INTERNAL-HOST"}}, 500)

        def ev(name: str, data: dict[str, Any]) -> bytes:
            return f"event: {name}\ndata: {json.dumps(data)}\n\n".encode()

        async def gen():
            yield ev("response.created", {"type": "response.created", "response": {"id": f"resp_mock_{call.n}"}})
            if mode == "big":
                for i in range(60):
                    yield ev("response.output_text.delta", {"delta": f"chunk{i}-" + "x" * 500})
                    await asyncio.sleep(0.005)
            else:
                for word in answer.split(" "):
                    yield ev("response.output_text.delta", {"delta": word + " "})
                    await asyncio.sleep(0.02)
            if mode == "drop":
                await asyncio.sleep(0.05)
                raise RuntimeError("mock upstream: cutting the connection mid-stream")
            if mode == "no_completed":
                return
            yield ev("response.completed", {"type": "response.completed",
                                            "response": {"id": f"resp_mock_{call.n}", "status": "completed"}})

        return StreamingResponse(gen(), media_type="text/event-stream")

    # ------------------------------------------------------------------ #
    async def start(self) -> None:
        config = uvicorn.Config(self._build_app(), host="127.0.0.1", port=self.port, log_level="error")
        self._server = uvicorn.Server(config)
        self._task = asyncio.create_task(self._server.serve())
        for _ in range(100):
            if self._server.started:
                c.out(f"mock upstream listening on {self.url}", 1)
                return
            await asyncio.sleep(0.05)
        raise RuntimeError("mock upstream failed to start (is the port in use? set MOCK_PORT)")

    async def stop(self) -> None:
        if self._server and self._task:
            self._server.should_exit = True
            try:
                await asyncio.wait_for(self._task, 5)
            except (asyncio.TimeoutError, Exception):  # noqa: BLE001
                self._task.cancel()
