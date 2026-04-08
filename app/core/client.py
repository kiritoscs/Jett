"""OCR client that manages all model services"""
from typing import Dict, Optional, List
from app.core.config import settings
from app.services.base import BaseOCRService
from app.services.ppocrv4_server_service import PPOCRv4ServerService
from app.services.ppocrv4_mobile_service import PPOCRv4MobileService
from app.services.easyocr_service import EasyOCRService
from app.services.rapidocr_service import RapidOCRService


class OCRClient:
    """OCR client that manages all available OCR services"""

    def __init__(self):
        self._services: Dict[str, BaseOCRService] = {}
        self._initialized = False

    def initialize(self) -> None:
        """Initialize all OCR services from configuration"""
        # PP-OCRv4 Server
        if settings.ppocrv4_server_endpoint:
            self._services["ppocrv4-server"] = PPOCRv4ServerService(settings.ppocrv4_server_endpoint)

        # PP-OCRv4 Mobile
        if settings.ppocrv4_mobile_endpoint:
            self._services["ppocrv4-mobile"] = PPOCRv4MobileService(settings.ppocrv4_mobile_endpoint)

        # EasyOCR
        if settings.easyocr_endpoint:
            self._services["easyocr"] = EasyOCRService(settings.easyocr_endpoint)

        # RapidOCR
        if settings.rapidocr_endpoint:
            self._services["rapidocr"] = RapidOCRService(settings.rapidocr_endpoint)

        self._initialized = True

    def get_service(self, model_id: str) -> Optional[BaseOCRService]:
        """Get OCR service by model ID"""
        if not self._initialized:
            self.initialize()
        return self._services.get(model_id)

    def list_available_models(self) -> List[Dict[str, str]]:
        """List all available models with metadata"""
        if not self._initialized:
            self.initialize()

        available = []
        for model_info in settings.available_models:
            if model_info["id"] in self._services:
                available.append(model_info)
        return available

    @property
    def services(self) -> Dict[str, BaseOCRService]:
        """Get all services"""
        if not self._initialized:
            self.initialize()
        return self._services


# Global singleton instance
ocr_client = OCRClient()
