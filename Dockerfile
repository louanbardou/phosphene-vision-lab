FROM python:3.11-slim

WORKDIR /app

# System deps needed by opencv-python-headless / nibabel at import time.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pipeline.py app.py ./
COPY config/ ./config/

EXPOSE 8501

# Model/data assets are downloaded on first use (public HuggingFace repos +
# NSD S3 bucket, no auth needed) and cached in assets_cache/ -- not baked
# into the image, since they run into the hundreds of MB.
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
