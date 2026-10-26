from typing import Any
from pydantic import BaseModel, ConfigDict


class ResponseRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    input: str | list[dict[str, Any]]
    stream: bool = False