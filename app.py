# create a server.py file

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import pipeline
import uvicorn
import asyncio
from asyncio import Lock
import os

app = FastAPI()

# Initialize only one model - we'll use chat by default
# This can be changed by the client by using the appropriate model_type
current_model = None
current_model_type = None
model_lock = Lock()

# Lock to ensure only one inference runs at a time
inference_lock = Lock()

class GenRequest(BaseModel):
    text: str
    max_new_tokens: int = 150
    do_sample: bool = False  # set True if you want to use temperature/top_p, etc.
    model_type: str = "chat"  # "summary" or "chat"

@app.get("/")
async def read_index():
    return FileResponse("index.html")

def load_model(model_type: str):
    """Load the specified model type"""
    global current_model, current_model_type
    if current_model_type != model_type:
        print(f"Loading {model_type} model...")
        current_model = pipeline("text-generation", model="Qwen/Qwen2.5-0.5B-Instruct")
        current_model_type = model_type
        print(f"{model_type} model loaded successfully")

@app.post("/generate")
async def generate(req: GenRequest):
    global current_model, current_model_type
    
    # Acquire lock to ensure only one inference at a time
    async with inference_lock:
        # Load model if needed (or switch models)
        async with model_lock:
            if current_model is None or current_model_type != req.model_type:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, load_model, req.model_type)
        
        # Run the inference in a thread pool to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        out = await loop.run_in_executor(
            None,
            lambda: current_model(
                req.text,
                max_new_tokens=req.max_new_tokens,
                do_sample=req.do_sample,
                truncation=True,
                return_full_text=False,
            )
        )
        return {"generated_text": out[0]["generated_text"]}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))  # HF sets PORT
    uvicorn.run("app:app", host="0.0.0.0", port=port)


# uvicorn server:app --host 0.0.0.0 --port 8000 --log-level debug


# curl -sS -i http://127.0.0.1:8000/generate \
#   -H "Content-Type: application/json" \
#   --data '{"text":"Hello","max_new_tokens":20}'

# curl -sS -i -L "https://<NAME>-8000.app.github.dev/generate" \
#   -H "Content-Type: application/json" \
#   --data '{"text":"What is the capital of France?","max_new_tokens":32}'


# This works on my machine :))
# curl -sS -i -L "https://verbose-space-lamp-697x5775rr5h7xg-8000.app.github.dev/generate" \
#   -H "Content-Type: application/json" \
#   --data '{"text":"What is the capital of France?","max_new_tokens":32}'