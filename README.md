# localinference

## Overview
`localinference` is a lightweight FastAPI web server that provides a simple UI and API for running text generation inference locally using the Qwen2.5-0.5B-Instruct model from Hugging Face. It features a modern, responsive web interface with real-time text generation capabilities, making it easy to interact with large language models directly from your browser.

## Features
- **Modern Web UI**: Beautiful, dark-themed interface with gradient effects and smooth animations for an enhanced user experience.
- **Interactive Text Generation**: Real-time text generation with loading states and error handling.
- **Model Type Selection**: Toggle between "summary" and "chat" model types (both currently use the same Qwen model, but the infrastructure supports different models).
- **Customizable Parameters**: Adjust maximum tokens and sampling behavior through the UI or API.
- **Thread-Safe Inference**: Asynchronous locks ensure only one inference runs at a time for resource efficiency.
- **RESTful API**: Clean JSON API endpoint for programmatic access and integration with other tools.

## Requirements
- Python 3.8+
- FastAPI
- Uvicorn
- Transformers
- Pydantic

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/jerwng/localinference.git
   cd localinference
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Starting the Server
1. Start the server:
   ```bash
   python server.py
   ```
   The server will run on `http://0.0.0.0:8000` by default.

2. Open your browser and navigate to:
   ```
   http://127.0.0.1:8000/
   ```
   You'll see the interactive web UI where you can:
   - Enter your input text
   - Toggle between model types (Summary/Chat)
   - Adjust the maximum number of tokens to generate
   - Click "Generate" to see the AI-generated response

### Using the API
You can also interact with the server programmatically using curl or any HTTP client:

```bash
curl -sS -i http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  --data '{"text":"Hello","max_new_tokens":20}'
```

## API Endpoints
### `GET /`
Serves the `index.html` file.

### `POST /generate`
Generates text based on the input parameters.
- **Request Body**:
  ```json
  {
    "text": "<input text>",
    "max_new_tokens": 150,
    "do_sample": false,
    "model_type": "chat"
  }
  ```
- **Response**:
  ```json
  {
    "generated_text": "<output text>"
  }
  ```

## Technical Details
- **Model**: Uses `Qwen/Qwen2.5-0.5B-Instruct`, a compact 0.5 billion parameter instruction-tuned model.
- **Framework**: Built with FastAPI for high-performance async API handling.
- **Frontend**: Pure HTML/CSS/JavaScript with no frameworks - lightweight and fast.
- **Concurrency**: Thread-safe inference with asyncio locks to prevent resource conflicts.
- **Auto-download**: The model will be automatically downloaded from Hugging Face on first run if not already cached locally.

## Notes
- First run may take some time as the model downloads (~500MB).
- The model runs on CPU by default. For GPU acceleration, ensure you have the appropriate PyTorch installation with CUDA support.
- Both "summary" and "chat" model types currently use the same Qwen model, but the code structure allows for easy expansion to support different models for each type.

