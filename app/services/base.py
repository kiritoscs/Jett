"""Base OCR service client"""
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Any


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
    async def process_file(self, file_bytes: bytes, filename: str, content_type: str) -> Tuple[str, Optional[float], Optional[Dict[str, Any]]]:
        """Process a raw file (image or PDF) and return extracted text, confidence, and raw result"""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the service is healthy"""
        pass
