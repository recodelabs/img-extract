import os
import uuid
import json
import base64
import io
import time
from datetime import datetime
from typing import Optional
from pathlib import Path
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import litellm
from PIL import Image
import aiofiles

load_dotenv()

app = FastAPI(title="Image Data Extraction API", version="1.0.0")

STORAGE_DIR = Path("storage")
IMAGES_DIR = STORAGE_DIR / "images"
LOGS_DIR = STORAGE_DIR / "logs"

@asynccontextmanager
async def lifespan(app: FastAPI):
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    yield

app = FastAPI(
    title="Image Data Extraction API", version="1.0.0", lifespan=lifespan
)

litellm.set_verbose = False

async def save_image_content(content: bytes, filename: str, file_id: str) -> str:
    file_extension = filename.split('.')[-1] if '.' in filename else 'jpg'
    new_filename = f"{file_id}.{file_extension}"
    file_path = IMAGES_DIR / new_filename
    
    async with aiofiles.open(file_path, 'wb') as f:
        await f.write(content)
    
    return str(file_path)

async def log_request_response(file_id: str, instructions: str, response: dict, image_path: str):
    timestamp = datetime.now().isoformat()
    log_data = {
        "file_id": file_id,
        "timestamp": timestamp,
        "instructions": instructions,
        "image_path": image_path,
        "response": response
    }
    
    log_filename = f"{file_id}_{timestamp.replace(':', '-')}.json"
    log_path = LOGS_DIR / log_filename
    
    async with aiofiles.open(log_path, 'w') as f:
        await f.write(json.dumps(log_data, indent=2))

@app.post("/extract")
async def extract_data_from_image(
    file: UploadFile = File(...),
    instructions: str = Form(...),
    model: Optional[str] = Form("gemini-2.0-flash-exp")
):
    file_id = str(uuid.uuid4())
    
    content = await file.read()
    if not content:
        error_response = {
            "success": False,
            "error": "The uploaded file is empty.",
            "file_id": file_id,
            "model_used": model,
            "timestamp": datetime.now().isoformat()
        }
        await log_request_response(file_id, instructions, error_response, "")
        return JSONResponse(content=error_response, status_code=400)

    try:
        Image.open(io.BytesIO(content))
    except Exception:
        error_response = {
            "success": False,
            "error": "The provided file is not a valid image.",
            "file_id": file_id,
            "model_used": model,
            "timestamp": datetime.now().isoformat()
        }
        await log_request_response(file_id, instructions, error_response, "")
        return JSONResponse(content=error_response, status_code=400)
    
    try:
        image_path = await save_image_content(content, file.filename, file_id)
        
        base64_image = base64.b64encode(content).decode('utf-8')
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Please analyze this image and follow these instructions: {instructions}. Return your response as structured data in JSON format."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ]
        
        llm_start_time = time.time()
        response = litellm.completion(
            model=model,
            messages=messages,
            max_tokens=1000,
            temperature=0.1
        )

        llm_end_time = time.time()
        llm_response_time = llm_end_time - llm_start_time
        
        extracted_data = response.choices[0].message.content
        
        # Extract token usage from response
        prompt_tokens = response.usage.prompt_tokens if response.usage else 0
        completion_tokens = response.usage.completion_tokens if response.usage else 0
        total_tokens = response.usage.total_tokens if response.usage else 0
        
        response_data = {
            "success": True,
            "file_id": file_id,
            "extracted_data": extracted_data,
            "model_used": model,
            "llm_response_time_seconds": round(llm_response_time, 3),
            "token_usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens
            },
            "timestamp": datetime.now().isoformat()
        }
        
        await log_request_response(file_id, instructions, response_data, image_path)
        
        return JSONResponse(content=response_data)
        
    except Exception as e:
        error_response = {
            "success": False,
            "error": str(e),
            "file_id": file_id,
            "model_used": model,
            "timestamp": datetime.now().isoformat()
        }
        await log_request_response(file_id, instructions, error_response, image_path if 'image_path' in locals() else "")
        return JSONResponse(content=error_response, status_code=500)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/")
async def root():
    return {
        "message": "Image Data Extraction API",
        "version": "1.0.0",
        "endpoints": {
            "POST /extract": "Extract data from image using LLM",
            "GET /health": "Health check",
            "GET /": "API information"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)