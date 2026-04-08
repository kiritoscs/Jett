"""PP-OCRv4 Server Serving"""
import os
import gc
import io
import logging
import time
import numpy as np
import cv2
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

from pydantic import BaseModel
from typing import List, Optional

# Set environment variables before importing paddle
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["CPU_NUM"] = "2"
os.environ["MKL_DEBUG_CPU_TYPE"] = "5"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="PP-OCRv4 Server Serving", version="0.1.0")


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


# Module-level model initialization (--preload ensures only Master runs this once)
try:
    logger.info("Initializing PaddleOCR model...")
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(
        use_angle_cls=True,
        lang="ch",
        device="cpu",
        show_log=False
    )
    logger.info("PaddleOCR model initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize PaddleOCR model: {e}")
    import traceback
    traceback.print_exc()
    ocr = None


@app.on_event("startup")
async def warmup():
    """Warmup model with dummy image"""
    global ocr
    try:
        logger.info("Warming up OCR model...")
        if ocr is not None:
            # Warmup with dummy image
            dummy_img = np.zeros((128, 128, 3), dtype=np.uint8)
            ocr.ocr(dummy_img, cls=True)
            logger.info("Model warmup completed successfully")
    except Exception as e:
        logger.error(f"Warmup failed: {e}")
        import traceback
        traceback.print_exc()


@app.get("/health")
async def health():
    """Health check"""
    if ocr is None:
        return JSONResponse(
            status_code=503,
            content={"success": False, "error": "Model not loaded"}
        )
    return {"status": "ok", "model": "ppocrv4-server"}


@app.get("/info")
async def info():
    """Model information"""
    return {
        "model_id": "ppocrv4-server",
        "name": "PP-OCRv4 Server",
        "description": "Full-sized PP-OCRv4 server version from PaddlePaddle",
        "loaded": ocr is not None
    }


@app.post("/ocr", response_model=OCRResponse)
async def ocr_api(file: UploadFile = File(...)):
    """Process OCR on uploaded image"""
    start_time = time.time()

    if ocr is None:
        processing_time = time.time() - start_time
        return OCRResponse(
            success=False,
            error="Model not loaded",
            processing_time=processing_time
        )

    try:
        # Read image
        image_data = await file.read()
        img = Image.open(io.BytesIO(image_data)).convert("RGB")

        # Resize if too large
        if max(img.size) > 2560:
            ratio = 2560 / max(img.size)
            img = img.resize((int(img.width * ratio), int(img.height * ratio)))

        # Convert to numpy array
        img_np = np.array(img)

        # Convert from RGB to BGR for OpenCV/PaddleOCR
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # Strict validation
        if img_np is None or img_np.size == 0:
            raise ValueError("Image reading failed or empty")
        if img_np.ndim != 3 or img_np.shape[-1] != 3:
            raise ValueError(f"Invalid image shape: {img_np.shape}, expected (H, W, 3)")

        logger.info(f"Processing image: shape={img_np.shape}, dtype={img_np.dtype}")

        # Run OCR
        result = ocr.ocr(img_np, cls=True)

        # Process result
        results = []
        full_text = []

        # PaddleOCR newer version returns: [line, line, ...] for single image
        lines = result[0] if result and isinstance(result[0], list) and len(result) == 1 else result

        if lines:
            for line in lines:
                if line is None:
                    continue
                bbox = line[0]  # [[x0,y0], [x1,y0], [x1,y1], [x0,y1]]
                text = line[1][0]
                confidence = line[1][1]

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

        # Clean up
        del img, img_np
        gc.collect()

        processing_time = time.time() - start_time

        return OCRResponse(
            success=True,
            text='\n'.join(full_text),
            results=results,
            processing_time=processing_time
        )

    except Exception as e:
        logger.error(f"OCR processing failed: {e}")
        import traceback
        traceback.print_exc()
        processing_time = time.time() - start_time
        return OCRResponse(
            success=False,
            error=str(e),
            processing_time=processing_time
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
