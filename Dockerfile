# syntax=docker/dockerfile:1
#
# Dockerfile for the zero-data cognitive model (v0.2.0).
# Runs the FastAPI service via uvicorn on port 8000.
#
# Base image: python:3.12-slim is used instead of 3.14 because official slim
# images for 3.14 may not be published yet; 3.12 is widely available and
# satisfies the package's requires-python (>=3.10). Slim (not alpine) is used
# because numpy/scipy ship manylinux wheels but would have to be built from
# source on musl-based alpine.

FROM python:3.12-slim

# Install curl so the compose healthcheck (`curl -f http://localhost:8000/`)
# can run inside the container. Keep this layer minimal; apt cache is purged
# in the same layer to avoid bloating the image.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Application working directory. All subsequent paths are relative to /app.
WORKDIR /app

# Copy only project metadata first so that the dependency-install layer is
# cached and not invalidated by source-code changes.
COPY pyproject.toml README.md LICENSE MANIFEST.in ./

# Copy the package source and the static web UI. tests/, examples/, docs/,
# benchmark scripts, etc. are excluded via .dockerignore.
COPY src/ ./src/
COPY web/ ./web/

# Install the package with the runtime extras needed by the API:
#   web     -> fastapi, uvicorn, pydantic
#   quantum -> qiskit
#   jit     -> numba
#   parallel-> joblib
# --no-cache-dir keeps the image smaller by not storing pip's wheel cache.
RUN pip install --no-cache-dir ".[web,quantum,jit,parallel]"

# Make the data directory available as a mount point for model persistence
# (compose mounts ./data:/app/data). Created here so the path exists even if
# no volume is attached.
RUN mkdir -p /app/data

# FastAPI / uvicorn listen here.
EXPOSE 8000

# Run the ASGI app. `zero_data_model.api:app` is the module-level FastAPI
# instance created by create_app() in src/zero_data_model/api.py.
CMD ["python", "-m", "uvicorn", "zero_data_model.api:app", "--host", "0.0.0.0", "--port", "8000"]
