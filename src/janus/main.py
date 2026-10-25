from fastapi import FastAPI


app = FastAPI()

@app.post("/v1/chat/completions")
async def forwardCompletion(item_id: int):
    data_type: str = str(type(item_id)) 
    return {"id": item_id, "type":data_type}