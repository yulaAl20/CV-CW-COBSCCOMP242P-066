
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

# libgl1 and libglib2.0-0 are what opencv-python-headless still links against.
# Without them `import cv2` fails at runtime with a confusing linker error.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first, so editing application code does not invalidate the
# layer that took two minutes to build.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/   ./backend/
COPY frontend/  ./frontend/
COPY artifacts/ ./artifacts/
COPY scripts/   ./scripts/
COPY samples/   ./samples/
COPY app.py .

# Model weights are mounted or downloaded at start-up, not baked in. Baking an
# 81 MB file into the image makes every rebuild slow and every push large.
RUN mkdir -p models

# Run as an unprivileged user. Hugging Face Spaces expects UID 1000.
RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,os; \
        urllib.request.urlopen(f'http://localhost:{os.environ.get(\"PORT\",8000)}/api/health')" \
        || exit 1


CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
