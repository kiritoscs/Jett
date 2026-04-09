"""RapidOCR Serving"""
import time
import io
import uuid
import base64
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
from pdf2image import convert_from_bytes

from pydantic import BaseModel
from typing import List, Optional, Dict, Any


app = FastAPI(title="RapidOCR Serving", version="0.1.0")

# Global model variable
model = None


class LayoutParsingRequest(BaseModel):
    """Layout parsing request with base64 file"""
    file: str  # base64 encoded
    fileType: int  # 0 = PDF, 1 = image


class UnifiedResponse(BaseModel):
    logId: str
    result: Optional[Dict[str, Any]] = None
    errorCode: int
    errorMsg: str


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


def process_single_image(image_np: np.ndarray, image_idx: int = 0):
    """Process a single image with OCR"""
    height, width = image_np.shape[:2]

    # Run OCR
    result, _ = model(image_np)

    # Process result into unified format
    parsing_res_list = []
    layout_det_boxes = []
    full_text = []

    if result:
        for idx, line in enumerate(result):
            bbox = line[0]  # [[x0,y0], [x1,y0], [x1,y1], [x0,y1]]
            text = line[1]
            confidence = line[2]

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

    if model is None:
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
        fileType = request.fileType  # 0 = PDF, 1 = image

        parsing_results = []
        all_text = []

        if fileType == 0:
            # PDF: convert to images and process each page
            try:
                images = convert_from_bytes(file_content, dpi=300)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to process PDF: {str(e)}")

            for i, img_pil in enumerate(images):
                image_np = np.array(img_pil)
                result = process_single_image(image_np, i)
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

        elif fileType == 1:
            # Image: process directly
            try:
                img_pil = Image.open(io.BytesIO(file_content)).convert('RGB')
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to open image: {str(e)}")

            image_np = np.array(img_pil)
            result = process_single_image(image_np, 0)
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
                errorMsg=f"Invalid fileType: {fileType}. Use 0 for PDF, 1 for image."
            )

        # Build unified result
        combined_text = "\n\n".join(all_text)

        result_data = {
            "layoutParsingResults": parsing_results,
            "dataInfo": {
                "width": parsing_results[0]["prunedResult"]["width"] if parsing_results else 0,
                "height": parsing_results[0]["prunedResult"]["height"] if parsing_results else 0,
                "type": "pdf" if fileType == 0 else "image"
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
        return UnifiedResponse(
            logId=log_id,
            result=None,
            errorCode=1,
            errorMsg=str(e)
        )


@app.post("/ocr", response_model=UnifiedResponse)
async def ocr_compat(file: UploadFile = File(...)):
    """Process OCR on uploaded image (compatibility endpoint)"""
    log_id = str(uuid.uuid4())

    if model is None:
        return UnifiedResponse(
            logId=log_id,
            result=None,
            errorCode=1,
            errorMsg="Model not loaded"
        )

    start_time = time.time()

    try:
        # Read file
        content = await file.read()
        filename = file.filename or ""

        # Determine file type
        fileType = 1  # default to image
        if filename.lower().endswith('.pdf') or (file.content_type and 'application/pdf' in file.content_type):
            fileType = 0

        # Encode to base64 and forward to layout-parsing
        import base64
        b64_content = base64.b64encode(content).decode('utf-8')

        # Call layout-parsing logic directly
        request = LayoutParsingRequest(file=b64_content, fileType=fileType)
        return await layout_parsing(request)

    except Exception as e:
        return UnifiedResponse(
            logId=log_id,
            result=None,
            errorCode=1,
            errorMsg=str(e)
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
