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

from pydantic import BaseModel, ValidationError
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import JSONResponse
import litellm
from PIL import Image
import aiofiles
import httpx

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

class JsonExtractRequest(BaseModel):
    instructions: str
    file_url: str
    model: Optional[str] = "gemini-2.0-flash-exp"

@app.post("/extract")
async def extract_data_from_image(request: Request):
    file_id = str(uuid.uuid4())
    content = None
    filename = "image.jpg"  # Default filename

    file: Optional[UploadFile] = None
    file_url: Optional[str] = None
    instructions: Optional[str] = None
    model: str = "gemini-2.0-flash-exp"

    content_type = request.headers.get("content-type", "")

    if "application/json" in content_type:
        try:
            body = await request.json()
            req_data = JsonExtractRequest(**body)
            instructions = req_data.instructions
            file_url = req_data.file_url
            model = req_data.model
        except ValidationError as e:
            return JSONResponse(
                status_code=422,
                content={"success": False, "error": f"Invalid JSON body: {e.errors()}"},
            )
        except json.JSONDecodeError:
            return JSONResponse(
                status_code=400, content={"success": False, "error": "Invalid JSON provided."}
            )

    elif "multipart/form-data" in content_type:
        form = await request.form()
        instructions = form.get("instructions")
        model = form.get("model", "gemini-2.0-flash-exp")
        file_from_form = form.get("file")
        if isinstance(file_from_form, UploadFile):
            file = file_from_form
        file_url = form.get("file_url")

    else:
        return JSONResponse(
            status_code=415,
            content={
                "success": False,
                "error": "Unsupported Content-Type. Please use 'application/json' or 'multipart/form-data'.",
            },
        )
    
    if not instructions:
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": "The 'instructions' field is required.",
            },
        )

    if file and file_url:
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": "Please provide either a file or a file_url, not both.",
            },
        )

    if not file and not file_url:
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": "Please provide a file or a file_url."},
        )

    if file:
        content = await file.read()
        if file.filename:
            filename = file.filename
    elif file_url:
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            async with httpx.AsyncClient() as client:
                response = await client.get(file_url, headers=headers)
                response.raise_for_status()
                content = await response.aread()
                # Try to get filename from URL
                parsed_url = httpx.URL(file_url)
                if parsed_url.path:
                    path_part = parsed_url.path.split("/")[-1]
                    if path_part:
                        filename = path_part

        except httpx.HTTPStatusError as e:
            error_response = {
                "success": False,
                "error": f"Failed to fetch image from URL: {e.response.status_code} {e.response.reason_phrase}",
                "file_id": file_id,
                "timestamp": datetime.now().isoformat(),
            }
            await log_request_response(file_id, instructions, error_response, "")
            return JSONResponse(content=error_response, status_code=400)
        except Exception as e:
            error_response = {
                "success": False,
                "error": f"An error occurred while fetching the image from URL: {str(e)}",
                "file_id": file_id,
                "timestamp": datetime.now().isoformat(),
            }
            await log_request_response(file_id, instructions, error_response, "")
            return JSONResponse(content=error_response, status_code=400)

    if not content:
        error_response = {
            "success": False,
            "error": "The uploaded file is empty or could not be fetched.",
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
        image_path = await save_image_content(content, filename, file_id)
        
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
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)