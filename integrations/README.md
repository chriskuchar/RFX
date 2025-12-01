# RFX Integration Examples

This folder contains integration examples for RFX with popular ML tools and deployment platforms.

## Contents

### 🐳 Docker Integration (`docker/`)

Docker containers for reproducible RFX deployments.

**Files:**
- `Dockerfile.gpu` - GPU-enabled RFX container (NVIDIA CUDA base)
- `Dockerfile.cpu` - CPU-only RFX container (lightweight)
- `docker-compose.yml` - Multi-container setup for both versions

**Pre-built Images (Docker Hub):**

```bash
# Pull and run GPU version
docker pull ckuchar/rfx-gpu
docker run --gpus all -it -v $(pwd):/workspace ckuchar/rfx-gpu

# Pull and run CPU version
docker pull ckuchar/rfx-cpu
docker run -it -v $(pwd):/workspace ckuchar/rfx-cpu
```

**Build Custom Images:**

```bash
cd integrations/docker

# GPU version
docker build -t rfx-gpu -f Dockerfile.gpu .
docker run --gpus all -it -v $(pwd):/workspace rfx-gpu

# CPU version
docker build -t rfx-cpu -f Dockerfile.cpu .
docker run -it -v $(pwd):/workspace rfx-cpu

# Or use docker-compose
docker-compose up -d rfx-gpu
docker-compose exec rfx-gpu python your_script.py
```

## Additional Integration Ideas

Future integration examples to add:
- **Kubernetes**: Deployment manifests for GPU/CPU pods
- **DVC**: Data Version Control integration
- **MLflow**: Experiment tracking and model management
- **Ray**: Distributed training across multiple GPUs
- **FastAPI**: Model serving API
- **Airflow**: Scheduled training pipelines

## Contributing

Have an integration example to share? Submit a PR to this branch!

## License

All integration examples are provided under the same MIT License as RFX.



