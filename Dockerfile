FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 ffmpeg && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
COPY datasets ./datasets
COPY web ./web
RUN python -m pip install --no-cache-dir .
EXPOSE 8000
CMD ["python", "-m", "visionguard", "serve", "--host", "0.0.0.0", "--port", "8000"]
