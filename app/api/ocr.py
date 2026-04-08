"""OCR API endpoints"""
import time
from typing import List
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from pdf2image import convert_from_bytes
import io

from app.core.client import ocr_client
from app.models.ocr import OCRCombinedResponse, OCRResponse
from app.core.config import settings


router = APIRouter(prefix="/api", tags=["ocr"])


@router.get("/models")
async def list_models():
    """List all available OCR models"""
    return ocr_client.list_available_models()


@router.post("/ocr/{model_id}", response_model=OCRCombinedResponse)
async def process_ocr(model_id: str, file: UploadFile = File(...)):
    """Process OCR on uploaded file (image or PDF)"""

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

    # Process based on file type
    content_type = file.content_type or ""
    filename = file.filename or ""

    if filename.lower().endswith('.pdf') or 'application/pdf' in content_type:
        # PDF: convert to images and process each page
        try:
            images = convert_from_bytes(content, dpi=300)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to process PDF: {str(e)}")

        page_results: List[OCRResponse] = []
        all_text = []

        for i, image in enumerate(images):
            try:
                text, avg_confidence = await service.process_image(image)
                page_results.append(OCRResponse(
                    success=True,
                    text=text,
                    processing_time=0.0
                ))
                all_text.append(f"--- 第 {i+1} 页 ---\n{text}")
            except Exception as e:
                page_results.append(OCRResponse(
                    success=False,
                    error=str(e),
                    processing_time=0.0
                ))

        combined_text = '\n\n'.join(all_text)
        processing_time = time.time() - start_time

        return OCRCombinedResponse(
            model_id=model_id,
            model_name=model_name,
            text=combined_text,
            pages=page_results,
            processing_time=processing_time
        )

    else:
        # Image: process directly
        try:
            image = Image.open(io.BytesIO(content)).convert('RGB')
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to open image: {str(e)}")

        try:
            text, avg_confidence = await service.process_image(image)
            processing_time = time.time() - start_time

            return OCRCombinedResponse(
                model_id=model_id,
                model_name=model_name,
                text=text,
                processing_time=processing_time
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
