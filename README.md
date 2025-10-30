# localinference

## Overview
`localinference` is a FastAPI-based server designed to provide text generation capabilities using pre-trained models from the Hugging Face Transformers library. It supports dynamic model loading and ensures thread-safe inference for efficient and reliable performance.

## Features
- **Dynamic Model Loading**: Load and switch between different model types dynamically.
- **Thread-Safe Inference**: Ensures only one inference runs at a time using asynchronous locks.
- **Customizable Parameters**: Supports user-defined parameters such as `max_new_tokens` and `do_sample` for text generation.
- **Static File Serving**: Serves an `index.html` file at the root endpoint.

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
1. Start the server:
   ```bash
   python server.py
   ```
   The server will run on `http://0.0.0.0:8000` by default.

2. Access the root endpoint to view the static `index.html` file:
   ```bash
   curl http://127.0.0.1:8000/
   ```

3. Use the `/generate` endpoint to generate text:
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

## Notes
- The default model used is `Qwen/Qwen2.5-0.5B-Instruct`.
- Ensure that the required model is available locally or can be downloaded from Hugging Face.

## License
This project is licensed under the MIT License.