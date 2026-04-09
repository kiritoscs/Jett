"""OCR API endpoints"""
import time
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from app.core.client import ocr_client
from app.models.ocr import OCRCombinedResponse
from app.core.config import settings


router = APIRouter(prefix="/api", tags=["ocr"])


@router.get("/models")
async def list_models():
    """List all available OCR models"""
    return ocr_client.list_available_models()


@router.post("/ocr/{model_id}", response_model=OCRCombinedResponse)
async def process_ocr(model_id: str, file: UploadFile = File(...)):
    """Process OCR on uploaded file (image or PDF) - FormData only for simplicity"""

    # Check model exists
    service = ocr_client.get_service(model_id)
    if service is None:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

    # Check file size
    if file.size and file.size > settings.max_file_size_mb * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {settings.max_file_size_mb}MB"
        )

    start_time = time.time()
    content = await file.read()

    # Get model info
    model_name = service.model_name
    content_type = file.content_type or ""
    filename = file.filename or ""

    try:
        # Use process_file to send raw file directly to backend service
        # This avoids memory issues with PDF processing in webui
        text, avg_confidence, raw_result = await service.process_file(
            content, filename, content_type
        )
        processing_time = time.time() - start_time

        return OCRCombinedResponse(
            model_id=model_id,
            model_name=model_name,
            text=text,
            processing_time=processing_time,
            raw_result=raw_result
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}")


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    healthy = True
    status = {}

    for model_id, service in ocr_client.services.items():
        try:
            is_healthy = await service.health_check()
            status[model_id] = "healthy" if is_healthy else "unhealthy"
            if not is_healthy:
                healthy = False
        except Exception:
            status[model_id] = "unreachable"
            healthy = False

    if healthy:
        return {"status": "ok", "models": status}
    else:
        return JSONResponse(
            status_code=503,
            content={"status": "degraded", "models": status}
        )
