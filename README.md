# Image Data Extraction API

A FastAPI server that uses LLMs to extract structured data from images based on custom instructions.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set your Gemini API key:
```bash
export GEMINI_API_KEY="your-api-key-here"
```

3. Run the server:
```bash
python main.py
```

The server will start on `http://localhost:8000`

## Usage

### Extract Data from Image

**POST** `/extract`

Form data:
- `file`: Image file (JPG, PNG, etc.)
- `instructions`: Text instructions for data extraction
- `model` (optional): LLM model to use (default: "gemini/gemini-2.0-flash-exp")

Example using curl:
```bash
curl -X POST "http://localhost:8000/extract" \
  -F "file=@drivers_license.jpg" \
  -F "instructions=Extract the driver's license number, name, date of birth, and expiration date from this image. Return as JSON."
```

### Health Check

**GET** `/health`

Returns server status and timestamp.

## Features

- **Image Storage**: Uploaded images are saved with unique IDs in `storage/images/`
- **Request Logging**: All requests and responses are logged in `storage/logs/` for debugging
- **Multiple Models**: Supports various LLM models via LiteLLM
- **Error Handling**: Comprehensive error handling with detailed logging

## File Structure

```
imgextract/
├── main.py              # Main FastAPI application
├── requirements.txt     # Python dependencies
├── storage/
│   ├── images/         # Stored uploaded images
│   └── logs/           # Request/response logs
└── README.md           # This file
```