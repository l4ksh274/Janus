import re
import hashlib


def normalise_prompt(prompt: str) -> str:
    prompt = prompt.strip().lower()
    re.sub(r"\s+", " ", prompt)

    return prompt

def gen_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()
