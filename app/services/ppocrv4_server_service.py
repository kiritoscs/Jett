"""PP-OCRv4 Server service client"""
import httpx
import base64
from typing import Tuple, Optional, Dict, Any
from .base import BaseOCRService


class PPOCRv4ServerService(BaseOCRService):
    """PP-OCRv4 Server service client"""

    def __init__(self, endpoint: str):
        self.endpoint = endpoint.rstrip('/')
        self.client = httpx.AsyncClient(timeout=120.0)

    @property
    def model_id(self) -> str:
        return "ppocrv4-server"

    @property
    def model_name(self) -> str:
        return "PP-OCRv4 Server"

    async def process_file(self, file_bytes: bytes, filename: str, content_type: str) -> Tuple[str, Optional[float], Optional[Dict[str, Any]]]:
        """Process raw file with PP-OCRv4 Server service using layout-parsing protocol"""
        b64_file = base64.b64encode(file_bytes).decode('utf-8')

        # Determine file type
        fileType = 1  # default to image
        if filename.lower().endswith('.pdf') or 'application/pdf' in content_type:
            fileType = 0

        # Send to layout-parsing endpoint
        response = await self.client.post(
            f"{self.endpoint}/layout-parsing",
            json={
                "file": b64_file,
                "fileType": fileType
            }
        )
        response.raise_for_status()
        result = response.json()

        # Parse unified response
        if result.get('errorCode') != 0:
            raise Exception(f"PP-OCRv4 Server processing failed: {result.get('errorMsg', 'Unknown error')}")

        # Extract full text from markdown or parsing results
        full_result = result.get('result', {})
        text = ''
        avg_confidence = None

        # Get text from layout parsing results
        layout_results = full_result.get('layoutParsingResults', [])
        if layout_results:
            # Combine all pages if PDF
            all_text = []
            all_confidences = []

            for page_result in layout_results:
                markdown = page_result.get('markdown', {})
                page_text = markdown.get('text', '') if markdown else ''

                # Also check parsing results if markdown is empty
                if not page_text:
                    pruned_result = page_result.get('prunedResult', {})
                    parsing_res = pruned_result.get('parsing_res_list', [])
                    texts = [r.get('block_content', '') for r in parsing_res]
                    page_text = '\n'.join(texts)

                if len(layout_results) > 1:
                    page_idx = layout_results.index(page_result) + 1
                    all_text.append(f"--- 第 {page_idx} 页 ---\n{page_text}")
                else:
                    all_text.append(page_text)

                # Calculate average confidence
                pruned_result = page_result.get('prunedResult', {})
                layout_det = pruned_result.get('layout_det_res', {}).get('boxes', [])
                if layout_det:
                    confidences = [r.get('score', 0) for r in layout_det if r.get('score') is not None]
                    all_confidences.extend(confidences)

            text = '\n\n'.join(all_text)
            if all_confidences:
                avg_confidence = sum(all_confidences) / len(all_confidences)

        return text, avg_confidence, result

    async def health_check(self) -> bool:
        """Check if PP-OCRv4 Server service is healthy"""
        try:
            response = await self.client.get(f"{self.endpoint}/health")
            return response.status_code == 200
        except Exception:
            return False

    async def __aexit__(self, exc_type, exc, tb):
        await self.client.aclose()
