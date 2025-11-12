# create a server.py file

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import uvicorn
import asyncio
from asyncio import Lock
import os

app = FastAPI()

# Initialize only one model - we'll use chat by default
# This can be changed by the client by using the appropriate model_type
current_model = None
current_tokenizer = None
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
    global current_model, current_tokenizer, current_model_type
    if current_model_type != model_type:
        print(f"Loading {model_type} model...")
        model_name = "Qwen/Qwen2.5-0.5B"
        current_tokenizer = AutoTokenizer.from_pretrained(model_name)
        current_model = AutoModelForCausalLM.from_pretrained(model_name)
        current_model.eval()
        current_model_type = model_type
        print(f"{model_type} model loaded successfully")

@app.post("/generate")
async def generate(req: GenRequest):
    global current_model, current_tokenizer, current_model_type
    
    # Acquire lock to ensure only one inference at a time
    async with inference_lock:
        # Load model if needed (or switch models)
        async with model_lock:
            if current_model is None or current_model_type != req.model_type:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, load_model, req.model_type)
        
        # Run the inference in a thread pool to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        
        # Use different generation methods based on model type
        if req.model_type == "chat":
            # For chat mode, include generation details for hover functionality
            result = await loop.run_in_executor(
                None,
                lambda: generate_with_alternatives(
                    current_model,
                    current_tokenizer,
                    req.text,
                    req.max_new_tokens
                )
            )
        else:
            # For summary mode, use efficient generation without alternatives
            result = await loop.run_in_executor(
                None,
                lambda: generate_simple(
                    current_model,
                    current_tokenizer,
                    req.text,
                    req.max_new_tokens
                )
            )
        return result

def generate_with_alternatives(model, tokenizer, prompt, max_new_tokens=15):
    """Generate text token by token, showing top-5 alternatives for each token"""
    
    # Tokenize to torch tensors
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"]
    attention_mask = enc.get("attention_mask", torch.ones_like(input_ids))
    
    k = 6  # show top-5 (chosen + 4 skipped)
    stop_on_eos = True
    
    generation_details = []
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # take logits for the last position
            next_logits = outputs.logits[:, -1, :]  # (batch=1, vocab)
            
            # get top-k candidates
            topk = torch.topk(next_logits, k=k, dim=-1)
            top_ids = topk.indices[0].tolist()  # length k
            chosen_id = top_ids[0]
            skipped_ids = top_ids[1:]  # next 4
            
            # decode tokens for display
            def dec(tid):
                s = tokenizer.decode([tid], skip_special_tokens=False)
                return s.replace("\n", "\\n")
            
            chosen_str = dec(chosen_id)
            skipped_strs = [dec(tid) for tid in skipped_ids]
            
            # Store generation details
            generation_details.append({
                "chosen": chosen_str,
                "alternatives": skipped_strs
            })
            
            # append chosen token for next step
            new_token = torch.tensor([[chosen_id]])
            input_ids = torch.cat([input_ids, new_token], dim=1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype)],
                dim=1
            )
            
            # optional early stop on EOS/special
            if stop_on_eos and chosen_id in tokenizer.all_special_ids:
                break
    
    # Generate final text
    final_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    generated_text = final_text[len(prompt):]  # Only the newly generated part
    
    return {
        "generated_text": generated_text,
        "generation_details": generation_details,
        "full_text": final_text
    }

def generate_simple(model, tokenizer, prompt, max_new_tokens=15):
    """Generate text efficiently without computing alternatives (for summary mode)"""
    
    # Tokenize to torch tensors
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"]
    attention_mask = enc.get("attention_mask", torch.ones_like(input_ids))
    
    stop_on_eos = True
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # take logits for the last position
            next_logits = outputs.logits[:, -1, :]  # (batch=1, vocab)
            
            # Simple greedy decoding - just get the best token (more efficient)
            chosen_id = torch.argmax(next_logits, dim=-1).item()
            
            # append chosen token for next step
            new_token = torch.tensor([[chosen_id]])
            input_ids = torch.cat([input_ids, new_token], dim=1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype)],
                dim=1
            )
            
            # optional early stop on EOS/special
            if stop_on_eos and chosen_id in tokenizer.all_special_ids:
                break
    
    # Generate final text
    final_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    generated_text = final_text[len(prompt):]  # Only the newly generated part
    
    return {
        "generated_text": generated_text,
        "full_text": final_text
    }

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