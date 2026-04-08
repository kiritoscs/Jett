# Jett - Multi-Model AI Services on Kubernetes

A deploy-on-demand AI model platform with web UI supporting multiple OCR models running on Kubernetes. Extensible for more model types in the future.

## Features

- 🌐 **Web Interface**: Upload files, get AI model results directly in the browser
- 🔌 **Multiple OCR Models**: Support for PaddleOCR (PP-OCRv4 Server & Mobile), EasyOCR, RapidOCR
- 📦 **Independent Deployment**: Each model runs in its own pod, scale independently
- 🚀 **Kubernetes Native**: Managed by Helm, proper health checks, resource management
- 📄 **PDF Support**: Automatically converts PDF pages to images for OCR
- 🔧 **uv Package Management**: Modern Python dependency management with uv

## Supported OCR Models

| Model | Description | Resource Requirements |
|-------|-------------|----------------------|
| **PP-OCRv4 Server** | Full-sized general purpose OCR from PaddlePaddle | ~2GB RAM, 2 cores |
| **PP-OCRv4 Mobile** | Lightweight PP-OCRv4 mobile version | ~1GB RAM, 1 core |
| **EasyOCR** | Multi-language OCR supporting 80+ languages | ~3GB RAM, 2 cores |
| **RapidOCR** | Lightning fast OCR with ONNX Runtime | ~1GB RAM, 1 core |

*(More model types can be added easily in the future)*

## Project Structure

```
.
├── app/                     # Web UI application
│   ├── core/               # Configuration and client
│   ├── models/             # Data models
│   ├── api/                # API endpoints
│   ├── services/           # Service clients
│   └── static/             # Frontend (HTML/CSS/JS)
├── helm/
│   └── jett/               # Helm Chart for Kubernetes deployment
│       ├── templates/      # All Kubernetes manifest templates
│       ├── Chart.yaml      # Chart metadata
│       └── values.yaml     # Configurable values
├── serving/                # Model serving code
│   ├── ppocrv4_mobile/     # PP-OCRv4 Mobile serving
│   ├── ppocrv4_server/     # PP-OCRv4 Server serving
│   ├── easyocr/            # EasyOCR serving
│   ├── rapidocr/           # RapidOCR serving
│   └── glm_ocr/            # GLM-OCR (deprecated)
├── Dockerfile.*            # Dockerfiles for each component
└── pyproject.toml          # Project dependencies
```

## Quick Start (Local Development)

### 1. Install dependencies with uv

```bash
uv sync
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
```

### 2. Run the web server locally

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Open the web interface

Visit `http://localhost:8000` in your browser.

**Note**: For local development without deploying models to Kubernetes, you can modify `app/core/config.py` to point to locally running model services.

## Deployment to Kubernetes

### 1. Build Docker images

```bash
# Use the build script (recommended)
python build_images.py v0.1.0

# Or build manually
docker build -t jett-webui:latest -f Dockerfile.webui .
docker build -t jett-ppocrv4-server:latest -f Dockerfile.ppocrv4-server .
docker build -t jett-ppocrv4-mobile:latest -f Dockerfile.ppocrv4-mobile .
docker build -t jett-easyocr:latest -f Dockerfile.easyocr .
docker build -t jett-rapidocr:latest -f Dockerfile.rapidocr .
```

If using a container registry, tag and push the images:
```bash
# Update registry.example.com to your actual registry
docker tag jett-webui:latest registry.example.com/jett-webui:latest
docker push registry.example.com/jett-webui:latest

docker tag jett-ppocrv4-server:latest registry.example.com/jett-ppocrv4-server:latest
docker push registry.example.com/jett-ppocrv4-server:latest

docker tag jett-ppocrv4-mobile:latest registry.example.com/jett-ppocrv4-mobile:latest
docker push registry.example.com/jett-ppocrv4-mobile:latest

docker tag jett-easyocr:latest registry.example.com/jett-easyocr:latest
docker push registry.example.com/jett-easyocr:latest

docker tag jett-rapidocr:latest registry.example.com/jett-rapidocr:latest
docker push registry.example.com/jett-rapidocr:latest
```
Then update the image repository in `helm/jett/values.yaml`.

### 2. Deploy with Helm

```bash
# First, edit helm/jett/values.yaml to:
# - Enable/disable components you want/don't want
# - Adjust resource allocations if needed
# - Update image registry and tags
# (ingress.host is already set to jett.com by default)

# Install the chart
helm install jett helm/jett --namespace jett --create-namespace

# Check deployment status
helm list -n jett
kubectl get pods -n jett
```

### 3. Access the web UI

**Via Ingress (Kong)**:
After deployment, the Ingress will be created automatically for domain `jett.com`. Update your DNS to point to your Kong ingress controller's external IP and access via `http://jett.com`.

**Via port-forward (for testing)**:
```bash
kubectl port-forward -n jett svc/webui 8000:80
```
Then open `http://localhost:8000` in your browser.

## Architecture

This project uses a **microservices architecture** where each AI model runs as an independent service:

```
┌─────────────┐
│   Browser   │
└──────┬──────┘
       │
       ▼
┌─────────────┐      ┌──────────────────┐
│   Web UI    │──────▶│PP-OCRv4 Server  │
│  (FastAPI)  │      └──────────────────┘
│             │      ┌──────────────────┐
│             │──────▶│PP-OCRv4 Mobile  │
│             │      └──────────────────┘
│             │      ┌──────────────────┐
│             │──────▶│     EasyOCR      │
│             │      └──────────────────┘
│             │      ┌──────────────────┐
│             │──────▶│    RapidOCR      │
└─────────────┘      └──────────────────┘
```

**Benefits of this architecture:**
- Each model can be scaled independently based on usage
- Models can be updated without affecting others
- Only deploy the models you actually need
- Better resource isolation
- Easy to add new model types in the future

## API Endpoints

### Web UI Backend API

- `GET /` - Web interface
- `GET /api/models` - List available models
- `POST /api/ocr/{model_id}` - Upload file and process OCR
- `GET /health` - Health check

### Each OCR Model Service API

- `POST /ocr` - Process image
- `GET /health` - Health check
- `GET /info` - Model information

## Adding a New Model

To add a new AI model:

1. Create serving code in `serving/{model_name}/main.py` following the existing pattern
2. Create a `Dockerfile.{model_name}`
3. Add the model to `helm/jett/values.yaml` for configuration
4. Add template files in `helm/jett/templates/` for deployment and service
5. Add the model to `app/core/config.py` in `available_models`
6. Create a service client in `app/services/{model_name}_service.py`
7. Update the client initialization in `app/core/client.py`

## Requirements

- Python 3.11+
- uv for dependency management
- Docker for building images
- Kubernetes cluster + Helm 3 for deployment
- Kong Ingress Controller (pre-configured in this chart)

## License

MIT
