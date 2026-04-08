"""Main FastAPI application"""
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

from app.api.ocr import router as ocr_router
from app.core.client import ocr_client

# Initialize OCR client on startup
ocr_client.initialize()

app = FastAPI(
    title="Jinx OCR",
    description="Multi-model OCR service with Kubernetes deployment",
    version="0.1.0"
)

# Include API routes
app.include_router(ocr_router)

# Mount static files
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/")
    async def root():
        """Serve the main HTML page"""
        return FileResponse(static_dir / "index.html")


@app.get("/health")
async def health():
    """Simple health check"""
    return {"status": "ok", "service": "webui"}
