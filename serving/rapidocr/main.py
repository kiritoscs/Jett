"""RapidOCR Serving"""
import time
import io
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np

from pydantic import BaseModel
from typing import List, Optional


app = FastAPI(title="RapidOCR Serving", version="0.1.0")

# Global model variable
model = None


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
    """Load RapidOCR model on startup"""
    global model
    try:
        from rapidocr_onnxruntime import RapidOCR
        # RapidOCR - uses ONNX Runtime for CPU inference, very fast
        model = RapidOCR()
        print("RapidOCR model loaded successfully")
    except Exception as e:
        print(f"Failed to load RapidOCR model: {e}")
        raise


@app.get("/health")
async def health():
    """Health check"""
    if model is None:
        return JSONResponse(
            status_code=503,
            content={"success": False, "error": "Model not loaded"}
        )
    return {"status": "ok", "model": "rapidocr"}


@app.get("/info")
async def info():
    """Model information"""
    return {
        "model_id": "rapidocr",
        "name": "RapidOCR",
        "description": "Lightning fast OCR based on ONNX Runtime, optimized for CPU",
        "loaded": model is not None
    }


@app.post("/ocr", response_model=OCRResponse)
async def ocr(file: UploadFile = File(...)):
    """Process OCR on uploaded image"""
    if model is None:
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
        image_np = np.array(image)

        # Run OCR
        result, _ = model(image_np)

        # Process result
        results = []
        full_text = []

        if result:
            for line in result:
                bbox = line[0]  # [[x0,y0], [x1,y0], [x1,y1], [x0,y1]]
                text = line[1]
                confidence = line[2]

                # Flatten bbox
                x0 = min(p[0] for p in bbox)
                y0 = min(p[1] for p in bbox)
                x1 = max(p[0] for p in bbox)
                y1 = max(p[1] for p in bbox)

                results.append(OCRResult(
                    text=text,
                    confidence=confidence,
                    bbox=BBox(x0=x0, y0=y0, x1=x1, y1=y1)
                ))
                full_text.append(text)

        processing_time = time.time() - start_time

        return OCRResponse(
            success=True,
            text='\n'.join(full_text),
            results=results,
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
