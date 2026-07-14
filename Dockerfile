FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    JAX_PLATFORMS=cpu \
    XLA_PYTHON_CLIENT_PREALLOCATE=false

WORKDIR /app

COPY requirements-server.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-server.txt

COPY gozero ./gozero

EXPOSE 8765

ENTRYPOINT ["python", "-m", "gozero.server"]
CMD ["--ckpt", "/models/latest.pkl", "--host", "0.0.0.0", "--port", "8765", "--state-file", "/data/app_games.json"]
