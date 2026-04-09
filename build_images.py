#!/usr/bin/env python3
"""
Build and push images to private registry with specified version tag.
Usage:
    python build_images.py <version>                          # Build all images
    python build_images.py <version> --service=<service-name>  # Build single service
Example:
    python build_images.py v0.0.1
    python build_images.py v0.0.5 --service=webui

Environment Variables:
    REGISTRY: Override default registry (default: middleware-docker:5001)
    IMAGE_PREFIX: Prefix for image names (default: jett-)
    REGISTRY_NAMESPACE: Namespace/path in registry (e.g., kiritoscs/jett)
"""

import sys
import subprocess
import argparse
import os
from pathlib import Path

import yaml

# Configuration
DEFAULT_REGISTRY = "middleware-docker:5001"
REGISTRY = os.environ.get("REGISTRY", DEFAULT_REGISTRY)
IMAGE_PREFIX = os.environ.get("IMAGE_PREFIX", "jett-")
REGISTRY_NAMESPACE = os.environ.get("REGISTRY_NAMESPACE", "")

# Build image name with optional namespace
def build_image_name(name: str) -> str:
    if REGISTRY_NAMESPACE:
        return f"{REGISTRY_NAMESPACE}/{name}"
    return name

IMAGES = [
    {
        "name": "jett-webui",
        "dockerfile": "Dockerfile.webui",
        "yaml_key": ["images", "webui", "tag"]
    },
    {
        "name": "jett-ppocrv4-server",
        "dockerfile": "Dockerfile.ppocrv4-server",
        "yaml_key": ["images", "ppocrv4Server", "tag"]
    },
    {
        "name": "jett-ppocrv4-mobile",
        "dockerfile": "Dockerfile.ppocrv4-mobile",
        "yaml_key": ["images", "ppocrv4Mobile", "tag"]
    },
    {
        "name": "jett-easyocr",
        "dockerfile": "Dockerfile.easyocr",
        "yaml_key": ["images", "easyocr", "tag"]
    },
    {
        "name": "jett-rapidocr",
        "dockerfile": "Dockerfile.rapidocr",
        "yaml_key": ["images", "rapidocr", "tag"]
    }
]
VALUES_FILE = Path("helm/jett/values.yaml")

SERVICE_NAME_MAP = {
    "webui": "jett-webui",
    "ppocrv4-server": "jett-ppocrv4-server",
    "ppocrv4server": "jett-ppocrv4-server",
    "ppocrv4-mobile": "jett-ppocrv4-mobile",
    "ppocrv4_mobile": "jett-ppocrv4-mobile",
    "easyocr": "jett-easyocr",
    "rapidocr": "jett-rapidocr",
}


def run_command(cmd: str) -> int:
    """Run a shell command and return exit code"""
    print(f"\n> {cmd}")
    return subprocess.run(cmd, shell=True).returncode


def image_exists(full_name: str) -> bool:
    """Check if image already exists locally"""
    result = subprocess.run(
        f"docker image inspect {full_name}",
        shell=True,
        capture_output=True,
        text=True
    )
    return result.returncode == 0


def build_push_image(name: str, dockerfile: str, version: str) -> bool:
    """Check if image exists, build and push if not exists"""
    image_name = build_image_name(name)
    full_name = f"{REGISTRY}/{image_name}:{version}"

    # Check if already exists
    if image_exists(full_name):
        print(f"✓ Image {full_name} already exists locally, skipping build/push")
        return True

    # Build
    print(f"→ Image {full_name} does not exist, building...")
    ret = run_command(f"docker build -t {full_name} -f {dockerfile} .")
    if ret != 0:
        print(f"ERROR: Failed to build {full_name}")
        return False

    # Push
    ret = run_command(f"docker push {full_name}")
    if ret != 0:
        print(f"ERROR: Failed to push {full_name}")
        return False

    print(f"✓ Successfully built and pushed {full_name}")
    return True


def update_values_yaml(version: str, target_image: dict = None) -> bool:
    """Update tags in values.yaml using pyyaml.
    If target_image is provided, only update that single image.
    """
    print(f"\n=> Updating {VALUES_FILE}")

    try:
        # Read and parse YAML
        with open(VALUES_FILE, 'r') as f:
            data = yaml.safe_load(f)

        images_to_update = [target_image] if target_image else IMAGES

        # Update each image tag
        for img in images_to_update:
            # Walk the nested dict
            current = data
            for key in img['yaml_key'][:-1]:
                current = current[key]
            # Update the tag
            current[img['yaml_key'][-1]] = version

        # Write back
        with open(VALUES_FILE, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

        if target_image:
            print(f"✓ Updated tag for {target_image['name']} to {version}")
        else:
            print(f"✓ Updated all tags in {VALUES_FILE}")
        return True
    except Exception as e:
        print(f"ERROR: Failed to update values.yaml: {e}")
        return False


def get_image_by_service_name(service_name: str):
    """Get image config by service name alias"""
    # Try direct match first
    for img in IMAGES:
        if img['name'] == service_name:
            return img

    # Try mapped name
    mapped_name = SERVICE_NAME_MAP.get(service_name)
    if mapped_name:
        for img in IMAGES:
            if img['name'] == mapped_name:
                return img

    return None


def main():
    parser = argparse.ArgumentParser(
        description='Build images, push to registry, update values.yaml'
    )
    parser.add_argument('version', help='Version tag (e.g. v0.0.1)')
    parser.add_argument('--service', '-s', help='Build only specific service (webui|ppocrv4-server|ppocrv4-mobile|easyocr|rapidocr)')
    args = parser.parse_args()

    # Check working directory
    if not Path('Dockerfile.webui').exists():
        print("ERROR: Must run this script from the project root directory")
        sys.exit(1)

    # Check if pyyaml is installed
    try:
        import yaml
    except ImportError:
        print("ERROR: pyyaml is not installed. Install with: uv pip install pyyaml")
        sys.exit(1)

    # Determine which images to process
    if args.service:
        target_img = get_image_by_service_name(args.service)
        if not target_img:
            available = ', '.join(['webui', 'ppocrv4-server', 'ppocrv4-mobile', 'easyocr', 'rapidocr'])
            print(f"ERROR: Unknown service '{args.service}'. Available: {available}")
            sys.exit(1)
        images_to_process = [target_img]
        print(f"=== Processing single service: {args.service} version {args.version} ===")
    else:
        images_to_process = IMAGES
        print(f"=== Processing all Jinx images version {args.version} ===")

    print(f"Registry: {REGISTRY}")
    print(f"Images: {', '.join(img['name'] for img in images_to_process)}\n")

    # Process selected images
    for img in images_to_process:
        print(f"\n--- Processing {img['name']}:{args.version} ---")
        success = build_push_image(img['name'], img['dockerfile'], args.version)
        if not success:
            print(f"\nFAILED at {img['name']}, aborting...")
            sys.exit(1)

    # Update values.yaml - only update the selected image if specified
    target_for_update = target_img if args.service else None
    success = update_values_yaml(args.version, target_for_update)
    if not success:
        sys.exit(1)

    # Generate output message
    lines = []
    for img in images_to_process:
        image_name = build_image_name(img['name'])
        full_name = f"{REGISTRY}/{image_name}:{args.version}"
        status = "exists locally, skipped" if image_exists(full_name) else "built & pushed"
        lines.append(f"  - {image_name}:{args.version} ({status})")

    if args.service:
        update_note = f"Updated: {VALUES_FILE} - tag for {target_img['name']} set to {args.version}"
    else:
        update_note = f"Updated: {VALUES_FILE} with all version tags"

    print(f"""
===========================================
✅ All done!

Processed:
{REGISTRY}:
{chr(10).join(lines)}

{update_note}

Now you can:
  helm upgrade jett helm/jett --namespace jett
===========================================
""")


if __name__ == "__main__":
    main()
