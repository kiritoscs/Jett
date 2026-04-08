"""OCR data models"""
from pydantic import BaseModel
from typing import List, Optional, Dict, Any


class BBox(BaseModel):
    """Bounding box for detected text"""
    x0: float
    y0: float
    x1: float
    y1: float


class OCRResult(BaseModel):
    """Single OCR result entry"""
    text: str
    confidence: float
    bbox: Optional[BBox] = None


class OCRResponse(BaseModel):
    """OCR response from model service"""
    success: bool
    text: Optional[str] = None
    results: Optional[List[OCRResult]] = None
    error: Optional[str] = None
    processing_time: float = 0.0


class OCRCombinedResponse(BaseModel):
    """Combined OCR response from web API"""
    model_id: str
    model_name: str
    text: str
    pages: Optional[List[OCRResponse]] = None
    processing_time: float = 0.0
