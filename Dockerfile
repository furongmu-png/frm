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

# Create a non-root user (uid 10001) so the container process never runs as
# root. The /app directory is chowned to this user so it can write model
# snapshots under /app/data (Fix 3, CWE-250).
RUN groupadd -r app && useradd -r -g app -u 10001 app

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
#   web     -> fastapi, uvicorn, pydantic, slowapi, structlog, prometheus
#   quantum -> qiskit
#   jit     -> numba
#   parallel-> joblib
# --no-cache-dir keeps the image smaller by not storing pip's wheel cache.
RUN pip install --no-cache-dir ".[web,quantum,jit,parallel]"

# Make the data directory available as a mount point for model persistence
# (compose mounts ./data:/app/data). Created here so the path exists even if
# no volume is attached. Owned by the non-root app user so /save can write.
RUN mkdir -p /app/data && chown -R app:app /app

# FastAPI / uvicorn listen here.
EXPOSE 8000

# Mark the runtime as production so docs/openapi are disabled (Q-MED-27).
ENV ZDM_ENV=production

# Drop privileges for the runtime process (Fix 3, CWE-250).
USER app

# Healthcheck uses Python stdlib only (no curl needed, S-LOW-09).
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=3).status==200 else 1)"

# Run the ASGI app. `zero_data_model.api:app` is the module-level FastAPI
# instance created by create_app() in src/zero_data_model/api.py.
# --timeout-graceful-shutdown 30 gives in-flight requests up to 30s to finish
# during a SIGTERM before uvicorn forces shutdown (Fix 4).
CMD ["python", "-m", "uvicorn", "zero_data_model.api:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--timeout-graceful-shutdown", "30"]
