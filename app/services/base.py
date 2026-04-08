"""Base OCR service client"""
from abc import ABC, abstractmethod
from typing import Tuple, Optional
from PIL import Image
import io


class BaseOCRService(ABC):
    """Base class for OCR service clients"""

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Model ID"""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Model display name"""
        pass

    @abstractmethod
    async def process_image(self, image: Image.Image) -> Tuple[str, Optional[float]]:
        """Process an image and return extracted text and confidence"""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the service is healthy"""
        pass


def image_to_bytes(image: Image.Image, format: str = "PNG") -> bytes:
    """Convert PIL Image to bytes"""
    buf = io.BytesIO()
    image.save(buf, format=format)
    buf.seek(0)
    return buf.getvalue()
