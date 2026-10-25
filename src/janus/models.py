from pydantic import BaseModel
from typing import List, Optional, Dict, Union

class Message(BaseModel):
    role: str
    content: Union[str, List[Dict[str, str]], None] # Referred to as input in OpenAI docs


class Response(BaseModel):
    model: str
    prompt: List[Message]

    



