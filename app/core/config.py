"""Configuration management"""
from pydantic_settings import BaseSettings
from typing import Dict


class Settings(BaseSettings):
    """Application settings"""

    # Service endpoints for each OCR model
    # These are set via environment variables, defaults work for same namespace K8s
    ppocrv4_server_endpoint: str = "http://ppocrv4-server:80"
    ppocrv4_mobile_endpoint: str = "http://ppocrv4-mobile:80"
    easyocr_endpoint: str = "http://easyocr:80"
    rapidocr_endpoint: str = "http://rapidocr:80"

    # Application settings
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    debug: bool = True

    # File upload settings
    max_file_size_mb: int = 10

    @property
    def model_endpoints(self) -> Dict[str, str]:
        """Get model endpoints mapping"""
        return {
            "ppocrv4-server": self.ppocrv4_server_endpoint,
            "ppocrv4-mobile": self.ppocrv4_mobile_endpoint,
            "easyocr": self.easyocr_endpoint,
            "rapidocr": self.rapidocr_endpoint,
        }

    @property
    def available_models(self) -> list[Dict[str, str]]:
        """Get list of available models"""
        models = [
            {
                "id": "ppocrv4-mobile",
                "name": "PP-OCRv4 Mobile",
                "description": "Lightweight PP-OCRv4 mobile version (~1GB RAM)"
            },
            {
                "id": "rapidocr",
                "name": "RapidOCR",
                "description": "Lightning fast OCR with ONNX Runtime (~1GB RAM)"
            },
            {
                "id": "ppocrv4-server",
                "name": "PP-OCRv4 Server",
                "description": "Full-sized PP-OCRv4 server version from PaddlePaddle (~2GB RAM)"
            },
            {
                "id": "easyocr",
                "name": "EasyOCR",
                "description": "Multi-language OCR supporting 80+ languages (~3GB RAM)"
            },
        ]
        return models

    class Config:
        env_prefix: str = "ocr_"
        case_sensitive: bool = False


settings = Settings()
