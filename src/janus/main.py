from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from janus.models import ResponseRequest
from httpx import AsyncClient

from dotenv import load_dotenv
from os import getenv


load_dotenv()

API_KEY = getenv("NVIDIA_API_KEY") or ""
BASE_URL = getenv("BASE_URL") or ""

if API_KEY == "" or BASE_URL == "":
    print("Environment variables haven't been set up")
    exit(1) 


app = FastAPI()
asyncClient = AsyncClient(headers={"Authorization": "Bearer " + API_KEY})

async def generate(responseRequest: ResponseRequest):
    async with asyncClient.stream("POST", BASE_URL + "/responses", json=responseRequest.model_dump(exclude_none=True)) as response :
        async for chunk in response.aiter_raw():
            yield chunk


@app.post("/v1/responses")
async def forwardCompletion(responseRequest: ResponseRequest):
    if responseRequest.stream:
        return StreamingResponse(generate(responseRequest), media_type="text/event-stream")
    else:
        response = await asyncClient.post(BASE_URL + "/responses", json=responseRequest.model_dump(exclude_none=True))
        return response.json()

        

