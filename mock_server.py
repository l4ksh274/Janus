"""Standalone mock upstream for load testing.

Starts a lightweight FastAPI server that mimics the NVIDIA Responses and
OpenAI-compatible Chat Completions APIs, returning deterministic responses so
the Janus proxy can populate its cache.

Usage:
    python mock_server.py              # defaults to port 9099
    MOCK_PORT=9099 python mock_server.py

Then set BASE_URL=http://127.0.0.1:9099/v1 in your .env and restart the proxy.
"""

from __future__ import annotations

import asyncio
import json
import os
import time

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

MOCK_PORT = int(os.getenv("MOCK_PORT", "9099"))
_call_count = 0

app = FastAPI(title="Janus Mock Upstream")


def _input_text(payload: dict, key: str) -> str:
    value = payload.get(key, "")
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


@app.post("/v1/responses")
async def responses(request: Request):
    global _call_count
    _call_count += 1
    n = _call_count

    payload = await request.json()
    text = _input_text(payload, "input")[:80]
    answer = f"MOCK-ANSWER#{n} for: {text[:60]}"

    # Simulate a small upstream latency (10-30 ms)
    await asyncio.sleep(0.01 + (n % 20) * 0.001)

    if payload.get("stream"):
        async def gen():
            yield _ev("response.created", {
                "type": "response.created",
                "response": {"id": f"resp_mock_{n}"},
            })
            for word in answer.split(" "):
                yield _ev("response.output_text.delta", {"delta": word + " "})
                await asyncio.sleep(0.005)
            yield _ev("response.completed", {
                "type": "response.completed",
                "response": {"id": f"resp_mock_{n}", "status": "completed"},
            })

        return StreamingResponse(gen(), media_type="text/event-stream")

    return JSONResponse({
        "id": f"resp_mock_{n}",
        "object": "response",
        "model": payload.get("model"),
        "status": "completed",
        "output": [{
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": answer}],
        }],
    })


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    global _call_count
    _call_count += 1
    n = _call_count

    payload = await request.json()
    text = _input_text(payload, "messages")[:80]
    answer = f"MOCK-ANSWER#{n} for: {text[:60]}"

    # Simulate the same small upstream latency as the Responses endpoint.
    await asyncio.sleep(0.01 + (n % 20) * 0.001)

    if payload.get("stream"):
        async def gen():
            created = int(time.time())
            chunk_id = f"chatcmpl_mock_{n}"
            yield _chat_chunk(chunk_id, created, payload.get("model"), {"role": "assistant"}, None)
            for word in answer.split(" "):
                yield _chat_chunk(
                    chunk_id, created, payload.get("model"), {"content": word + " "}, None
                )
                await asyncio.sleep(0.005)
            yield _chat_chunk(chunk_id, created, payload.get("model"), {}, "stop")
            yield b"data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    return JSONResponse({
        "id": f"chatcmpl_mock_{n}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": payload.get("model"),
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": answer},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": 1,
            "completion_tokens": len(answer.split()),
            "total_tokens": 1 + len(answer.split()),
        },
    })


def _chat_chunk(
    chunk_id: str,
    created: int,
    model: str | None,
    delta: dict,
    finish_reason: str | None,
) -> bytes:
    chunk = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(chunk)}\n\n".encode()


def _ev(name: str, data: dict) -> bytes:
    return f"event: {name}\ndata: {json.dumps(data)}\n\n".encode()


if __name__ == "__main__":
    print(f"Mock upstream starting on http://127.0.0.1:{MOCK_PORT}/v1")
    uvicorn.run(app, host="127.0.0.1", port=MOCK_PORT, log_level="warning")
