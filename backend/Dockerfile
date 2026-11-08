# Build from the REPOSITORY ROOT (the image needs the training/ preprocessing
# module to unpickle the model):
#     docker build -f backend/Dockerfile -t heartguard-api .
#     docker run -p 8000:8000 heartguard-api
#
# Python/scikit-learn versions must match the environment that trained the model.
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app

COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# Same relative layout as the repo, so the app finds the model and training code.
COPY backend/app backend/app
COPY training/src/__init__.py training/src/__init__.py
COPY training/src/features training/src/features
COPY training/artifacts/models training/artifacts/models
COPY docs/MODEL_CARD.md docs/MODEL_CARD.md

RUN useradd --create-home appuser
USER appuser

WORKDIR /app/backend
ENV PORT=8000
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import os,urllib.request as u; u.urlopen('http://localhost:%s/api/v1/health' % os.environ.get('PORT','8000'), timeout=4)" || exit 1
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers --forwarded-allow-ips='*'"]
