"""PP-OCRv4 Mobile Serving"""
import os
import gc
import io
import logging
import time
import uuid
import base64
import numpy as np
import cv2
from PIL import Image
from pdf2image import convert_from_bytes
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from pydantic import BaseModel
from typing import List, Optional, Dict, Any

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

app = FastAPI(title="PP-OCRv4 Mobile Serving", version="0.1.0")


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


class LayoutParsingRequest(BaseModel):
    """Layout parsing request with base64 file"""
    file: str  # base64 encoded
    filetype: int  # 0 = PDF, 1 = image


class UnifiedResponse(BaseModel):
    logId: str
    result: Optional[Dict[str, Any]] = None
    errorCode: int
    errorMsg: str


# Module-level model initialization (--preload ensures only Master runs this once)
try:
    logger.info("Initializing PaddleOCR model...")
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(
        use_angle_cls=True,
        lang="ch",
        det_model_dir=None,
        rec_model_dir=None,
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
    return {"status": "ok", "model": "ppocrv4-mobile"}


@app.get("/info")
async def info():
    """Model information"""
    return {
        "model_id": "ppocrv4-mobile",
        "name": "PP-OCRv4 Mobile",
        "description": "Lightweight PP-OCRv4 mobile version",
        "loaded": ocr is not None
    }


def process_single_image(image_np: np.ndarray, image_idx: int = 0):
    """Process a single image with OCR"""
    height, width = image_np.shape[:2]

    # Run OCR
    result = ocr.ocr(image_np, cls=True)

    # Process result into unified format
    parsing_res_list = []
    layout_det_boxes = []
    full_text = []

    # PaddleOCR newer version returns: [line, line, ...] for single image
    lines = result[0] if result and isinstance(result[0], list) and len(result) == 1 else result

    if lines:
        for idx, line in enumerate(lines):
            if line is None:
                continue
            bbox = line[0]  # [[x0,y0], [x1,y0], [x1,y1], [x0,y1]]
            text = line[1][0]
            confidence = line[1][1]

            # Flatten bbox to [x0, y0, x1, y1]
            x0 = min(p[0] for p in bbox)
            y0 = min(p[1] for p in bbox)
            x1 = max(p[0] for p in bbox)
            y1 = max(p[1] for p in bbox)
            flat_bbox = [int(x0), int(y0), int(x1), int(y1)]

            # Convert to polygon points
            polygon_points = [[float(p[0]), float(p[1])] for p in bbox]

            block_id = idx
            parsing_res_list.append({
                "block_label": "text",
                "block_content": text,
                "block_bbox": flat_bbox,
                "block_id": block_id,
                "block_order": idx,
                "group_id": idx,
                "block_polygon_points": polygon_points
            })

            layout_det_boxes.append({
                "cls_id": 22,
                "label": "text",
                "score": float(confidence),
                "coordinate": flat_bbox,
                "order": idx,
                "polygon_points": polygon_points
            })

            full_text.append(text)

    return {
        "width": width,
        "height": height,
        "parsing_res_list": parsing_res_list,
        "layout_det_boxes": layout_det_boxes,
        "full_text": full_text
    }


@app.post("/layout-parsing", response_model=UnifiedResponse)
async def layout_parsing(request: LayoutParsingRequest):
    """Process OCR with layout parsing using base64 encoded file"""
    log_id = str(uuid.uuid4())

    if ocr is None:
        return UnifiedResponse(
            logId=log_id,
            result=None,
            errorCode=1,
            errorMsg="Model not loaded"
        )

    start_time = time.time()

    try:
        # Decode base64 file
        file_content = base64.b64decode(request.file)

        # Process based on file type
        filetype = request.filetype  # 0 = PDF, 1 = image

        parsing_results = []
        all_text = []

        if filetype == 0:
            # PDF: convert to images and process each page
            try:
                images = convert_from_bytes(file_content, dpi=300)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to process PDF: {str(e)}")

            for i, img_pil in enumerate(images):
                # Convert to numpy array
                img_np = np.array(img_pil)
                # Convert from RGB to BGR for OpenCV/PaddleOCR
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

                result = process_single_image(img_np, i)
                markdown_text = "\n".join(result["full_text"])

                parsing_results.append({
                    "prunedResult": {
                        "page_count": len(images),
                        "width": result["width"],
                        "height": result["height"],
                        "model_settings": {
                            "use_doc_preprocessor": False,
                            "use_layout_detection": True,
                            "use_chart_recognition": False,
                            "use_seal_recognition": False,
                            "use_ocr_for_image_block": False,
                            "format_block_content": False,
                            "merge_layout_blocks": True,
                            "markdown_ignore_labels": [
                                "number", "footnote", "header", "header_image",
                                "footer", "footer_image", "aside_text"
                            ],
                            "return_layout_polygon_points": True
                        },
                        "parsing_res_list": result["parsing_res_list"],
                        "layout_det_res": {
                            "boxes": result["layout_det_boxes"]
                        }
                    },
                    "markdown": {
                        "text": markdown_text,
                        "images": {}
                    }
                })
                all_text.append(f"--- 第 {i+1} 页 ---\n{markdown_text}")

        elif filetype == 1:
            # Image: process directly
            try:
                img_pil = Image.open(io.BytesIO(file_content)).convert('RGB')
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to open image: {str(e)}")

            # Convert to numpy array
            img_np = np.array(img_pil)
            # Convert from RGB to BGR for OpenCV/PaddleOCR
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            result = process_single_image(img_np, 0)
            markdown_text = "\n".join(result["full_text"])

            parsing_results.append({
                "prunedResult": {
                    "page_count": None,
                    "width": result["width"],
                    "height": result["height"],
                    "model_settings": {
                        "use_doc_preprocessor": False,
                        "use_layout_detection": True,
                        "use_chart_recognition": False,
                        "use_seal_recognition": False,
                        "use_ocr_for_image_block": False,
                        "format_block_content": False,
                        "merge_layout_blocks": True,
                        "markdown_ignore_labels": [
                            "number", "footnote", "header", "header_image",
                            "footer", "footer_image", "aside_text"
                        ],
                        "return_layout_polygon_points": True
                    },
                    "parsing_res_list": result["parsing_res_list"],
                    "layout_det_res": {
                        "boxes": result["layout_det_boxes"]
                    }
                },
                "markdown": {
                    "text": markdown_text,
                    "images": {}
                }
            })
            all_text.append(markdown_text)

        else:
            return UnifiedResponse(
                logId=log_id,
                result=None,
                errorCode=1,
                errorMsg=f"Invalid filetype: {filetype}. Use 0 for PDF, 1 for image."
            )

        # Build unified result
        combined_text = "\n\n".join(all_text)

        result_data = {
            "layoutParsingResults": parsing_results,
            "dataInfo": {
                "width": parsing_results[0]["prunedResult"]["width"] if parsing_results else 0,
                "height": parsing_results[0]["prunedResult"]["height"] if parsing_results else 0,
                "type": "pdf" if filetype == 0 else "image"
            }
        }

        return UnifiedResponse(
            logId=log_id,
            result=result_data,
            errorCode=0,
            errorMsg="Success"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Layout parsing failed: {e}")
        import traceback
        traceback.print_exc()
        return UnifiedResponse(
            logId=log_id,
            result=None,
            errorCode=1,
            errorMsg=str(e)
        )


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
