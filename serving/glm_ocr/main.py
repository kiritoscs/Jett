"""GLM-OCR Serving - SGLang Inference"""
import os
import time
import io
import torch
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image

from pydantic import BaseModel
from typing import List, Optional

# HF_ENDPOINT for mirror site (https://hf-mirror.com) if needed
if "HF_ENDPOINT" in os.environ:
    os.environ["HF_ENDPOINT"] = os.environ["HF_ENDPOINT"]

app = FastAPI(title="GLM-OCR Serving", version="0.1.0")

# Global model and processor variables
model = None
processor = None
device = "cuda" if torch.cuda.is_available() else "cpu"


class BBox(BaseModel):
    x0: float
    y0: float
    x1: float
    y1: float


class OCRResult(BaseModel):
    text: str
    confidence: float
    bbox: Optional[BBox] = None


class OCRResponse(BaseModel):
    success: bool
    text: Optional[str] = None
    results: Optional[List[OCRResult]] = None
    error: Optional[str] = None
    processing_time: float


@app.on_event("startup")
async def load_model():
    """Load GLM-OCR model on startup from HuggingFace using SGLang"""
    global model, processor
    try:
        print(f"Loading GLM-OCR model on {device}...")
        print(f"Using HF endpoint: {os.environ.get('HF_ENDPOINT', 'https://huggingface.co')}")
        model_name = "zai-org/GLM-OCR"

        # Import SGLang
        from sglang import Engine
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        # Create SGLang engine
        model = Engine(
            model_path=model_name,
            trust_remote_code=True,
            device=device,
            dtype="bfloat16" if device == "cuda" else "float32",
        )

        print("GLM-OCR model loaded successfully with SGLang")
    except Exception as e:
        print(f"Failed to load GLM-OCR model: {e}")
        raise


@app.get("/health")
async def health():
    """Health check"""
    if model is None or processor is None:
        return JSONResponse(
            status_code=503,
            content={"success": False, "error": "Model not loaded"}
        )
    return {"status": "ok", "model": "glm-ocr"}


@app.get("/info")
async def info():
    """Model information"""
    return {
        "model_id": "glm-ocr",
        "name": "GLM-OCR",
        "description": "GLM based OCR model from zai-org, accelerated by SGLang",
        "device": device,
        "inference": "sglang",
        "loaded": model is not None and processor is not None
    }


@app.post("/ocr", response_model=OCRResponse)
async def ocr(file: UploadFile = File(...)):
    """Process OCR on uploaded image using GLM-OCR with SGLang"""
    if model is None or processor is None:
        return OCRResponse(
            success=False,
            error="Model not loaded",
            processing_time=0
        )

    start_time = time.time()

    try:
        # Read image
        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert('RGB')

        # Prepare inputs for GLM-OCR
        # GLM-OCR uses <OCR> token to trigger OCR
        prompt = "<OCR>"
        inputs = processor(
            images=image,
            text=prompt,
            return_tensors="pt",
            padding=True
        )

        # Move to device
        input_ids = inputs["input_ids"].to(device)
        pixel_values = inputs["pixel_values"].to(device)

        # Generate with SGLang
        output_ids = model.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            max_new_tokens=2048,
            temperature=0.0,
            do_sample=False
        )

        # Decode
        text = processor.decode(output_ids[0], skip_special_tokens=True)

        # Remove the prompt <OCR> from output
        text = text.replace("<OCR>", "").strip()

        processing_time = time.time() - start_time

        return OCRResponse(
            success=True,
            text=text,
            results=None,
            processing_time=processing_time
        )

    except Exception as e:
        processing_time = time.time() - start_time
        return OCRResponse(
            success=False,
            error=str(e),
            processing_time=processing_time
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
