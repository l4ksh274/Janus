from pydantic import BaseModel
from typing import List, Dict, Union, Any, Literal


class Message(BaseModel):
    role: Literal["system", "developer", "user", "assistant"]
    content: Union[str, List[Dict[str, Any]], None]

class CompletionRequest(BaseModel):
    model: str
    input: List[Message]
    tools: List[Dict[str, Any]] | None = None
    stream: bool = False


