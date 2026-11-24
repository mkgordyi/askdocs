FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ASKDOCS_INDEX_DIR=/srv/index \
    ASKDOCS_MODEL_CACHE=/srv/models

# Do not run the service as root. A fixed uid keeps file ownership predictable on
# every host (Hugging Face Spaces, for one, always runs containers as uid 1000).
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /srv \
    && chown appuser /srv
WORKDIR /srv

# Dependencies first: this layer is rebuilt only when requirements.txt changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=appuser askdocs ./askdocs
COPY --chown=appuser data ./data

USER appuser

# Build the search index into the image. The container starts with everything it
# needs and never downloads anything at runtime.
RUN python -m askdocs.index
ENV HF_HUB_OFFLINE=1

EXPOSE 8080
CMD ["uvicorn", "askdocs.api:app", "--host", "0.0.0.0", "--port", "8080"]
