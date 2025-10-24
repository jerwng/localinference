# create a server.py file

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
import uvicorn

app = FastAPI()
pipe = pipeline("text-generation", model="Qwen/Qwen2.5-0.5B-Instruct")

class GenRequest(BaseModel):
    text: str
    max_new_tokens: int = 150
    do_sample: bool = False  # set True if you want to use temperature/top_p, etc.

@app.post("/generate")
def generate(req: GenRequest):
    out = pipe(
        req.text,
        max_new_tokens=req.max_new_tokens,
        do_sample=req.do_sample,
        truncation=True,
        return_full_text=False,
    )
    return {"generated_text": out[0]["generated_text"]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


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