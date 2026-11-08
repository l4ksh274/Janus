"""Request models accepted by the proxy's upstream-facing routes.

These models mirror the NVIDIA/OpenAI payloads that Janus accepts before it
checks the Redis cache and forwards requests upstream.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict


class ResponseRequest(BaseModel):
    """Request body for the Responses API endpoint.

    The `input` payload can be a plain string or a structured message list, and
    callers may opt into streaming responses.
    """

    model_config = ConfigDict(extra="allow")

    model: str
    input: str | list[dict[str, Any]]
    stream: bool = False


class ChatCompletionRequest(BaseModel):
    """Request body for the OpenAI-compatible chat-completions endpoint."""

    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[dict[str, Any]]
    stream: bool = False
