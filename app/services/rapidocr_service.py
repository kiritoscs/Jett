"""RapidOCR service client"""
import httpx
from PIL import Image
from typing import Tuple, Optional
from .base import BaseOCRService, image_to_bytes


class RapidOCRService(BaseOCRService):
    """RapidOCR service client"""

    def __init__(self, endpoint: str):
        self.endpoint = endpoint.rstrip('/')
        self.client = httpx.AsyncClient(timeout=120.0)

    @property
    def model_id(self) -> str:
        return "rapidocr"

    @property
    def model_name(self) -> str:
        return "RapidOCR"

    async def process_image(self, image: Image.Image) -> Tuple[str, Optional[float]]:
        """Process image with RapidOCR service"""
        image_bytes = image_to_bytes(image)

        files = {'file': ('image.png', image_bytes, 'image/png')}

        response = await self.client.post(
            f"{self.endpoint}/ocr",
            files=files
        )
        response.raise_for_status()
        result = response.json()

        if result.get('success'):
            text = result.get('text', '')
            if not text and result.get('results'):
                # If results are available, join them
                text = '\n'.join([r.get('text', '') for r in result['results']])
            avg_confidence = None
            if result.get('results'):
                confidences = [r.get('confidence', 0) for r in result['results'] if r.get('confidence')]
                if confidences:
                    avg_confidence = sum(confidences) / len(confidences)
            return text, avg_confidence
        else:
            error = result.get('error', 'Unknown error')
            raise Exception(f"RapidOCR processing failed: {error}")

    async def health_check(self) -> bool:
        """Check if RapidOCR service is healthy"""
        try:
            response = await self.client.get(f"{self.endpoint}/health")
            return response.status_code == 200
        except Exception:
            return False

    async def __aexit__(self, exc_type, exc, tb):
        await self.client.aclose()
