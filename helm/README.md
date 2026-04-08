# Jinx Helm Chart

Helm Chart for deploying Jinx - Multi-model AI services with web UI on Kubernetes.

## Overview

This Helm Chart deploys:
- **webui**: Web interface and API gateway
- **paddleocr**: PaddleOCR full model service
- **glm-ocr**: GLM-OCR 0.9B model service (THUDM)
- **ppocrv4-mobile**: PP-OCRv4 Mobile lightweight model service

*(More model types can be added in the future)*

## Prerequisites

- Kubernetes 1.19+
- Helm 3.0+
- Kong Ingress Controller (already configured for your cluster)
- Docker images built (or accessible from your registry)

## Installing the Chart

```bash
# Clone or navigate to this project directory
cd /path/to/jett

# Install the chart
helm install jett helm/jett \
  --namespace jett \
  --create-namespace
```

## Install with custom values

```bash
# Edit values.yaml first
vim helm/jett/values.yaml

# Then install
helm install jett helm/jett --namespace jett --create-namespace
```

Or pass values directly:
```bash
helm install jett helm/jett \
  --namespace jett \
  --create-namespace \
  --set ingress.host=jett.com \
  --set components.glmOcr.enabled=false
```

## Uninstalling the Chart

```bash
helm uninstall jett --namespace jett
```

This removes all resources associated with the chart.

## Configuration

### Enable/disable components

You can enable or disable individual components in `values.yaml`:

```yaml
components:
  webui:
    enabled: true
  paddleocr:
    enabled: true
  glmOcr:
    enabled: true
  ppocrv4Mobile:
    enabled: true
```

Set `enabled: false` for models you don't want to deploy.

### Ingress

The chart is pre-configured for **Kong Ingress Controller**:

```yaml
ingress:
  enabled: true
  ingressClassName: "kong"
  host: "ocr.example.com"  # CHANGE THIS to your domain
```

### Resource Allocation

Default resource allocations:

| Component | CPU Request | Memory Request | CPU Limit | Memory Limit |
|-----------|-------------|----------------|-----------|--------------|
| Web UI | 250m | 256Mi | 500m | 512Mi |
| PaddleOCR | 1000m (1 core) | 2Gi | 2000m (2 cores) | 4Gi |
| GLM-OCR 0.9B | 2000m (2 cores) | 4Gi | 4000m (4 cores) | 8Gi |
| PP-OCRv4 Mobile | 500m | 1Gi | 1000m (1 core) | 2Gi |

Change these in `values.yaml` under the `resources` section.

### GPU Support

To enable GPU, uncomment the GPU line in resources:

```yaml
resources:
  glmOcr:
    limits:
      nvidia.com/gpu: "1"  # Uncomment this
```

Do the same for other models if you have GPU available.

### Images

By default, images are pulled locally (`pullPolicy: IfNotPresent`). If you're using a remote registry, update the repository:

```yaml
images:
  webui:
    repository: your-registry/jett-ocr-webui
    tag: latest
    pullPolicy: IfNotPresent
  # ... update for other images
```

## Commands

```bash
# Upgrade after changing values
helm upgrade jett helm/jett --namespace jett

# Check status
helm status jett --namespace jett

# View history
helm history jett --namespace jett

# Get all resources
kubectl get all -n jett
```

## Current Kong Ingress Configuration

This chart is already configured for your cluster with:
- `ingressClassName: "kong"`
- Kong-specific annotations already added
- Default domain: `jett.com`
- Your cluster already has `kong` IngressClass installed

Just update DNS and it will work.
