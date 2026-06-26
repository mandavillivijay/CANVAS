# syntax=docker/dockerfile:1
# canvas-heal server image
# Target: < 2 GB image size
# Published as: ghcr.io/mandavillivijay/canvas-heal:latest

FROM python:3.11-slim AS builder

WORKDIR /build

# Install build deps
RUN pip install --no-cache-dir --upgrade pip wheel

# Copy only what pip needs to install the package
COPY pyproject.toml README.md LICENSE ./
COPY canvas_heal/ canvas_heal/
COPY hatch_build.py ./

# Install the package + server extras into a target dir for copying
RUN pip install --no-cache-dir --prefix=/install .[server]

# ---------------------------------------------------------------------------
# Final image
# ---------------------------------------------------------------------------
FROM python:3.11-slim

# Sentence-transformers caches models here; mount as a volume in production
# so the 90 MB model is not re-downloaded on every container start.
ENV CANVAS_MODEL_CACHE_DIR=/var/cache/canvas-heal
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Expose default port
EXPOSE 8000

# Entrypoint: canvas-heal serve
# Override CANVAS_API_KEY, CANVAS_STORE_URL, and CANVAS_MODEL_CACHE_DIR via env.
ENTRYPOINT ["canvas-heal", "serve"]
CMD ["--host", "0.0.0.0", "--port", "8000", "--store-url", "sqlite:////data/canvas_intents.db"]
