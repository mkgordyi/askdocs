"""HTTP API. Thin on purpose: all decisions live in service.py."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from . import __version__
from .config import settings
from .generate import make_generator
from .index import load
from .service import AskService

logging.basicConfig(level=logging.INFO, format="%(message)s")
STATIC = Path(__file__).parent / "static"


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)


class Source(BaseModel):
    ref: str
    title: str
    url: str
    excerpt: str


class AskResponse(BaseModel):
    answered: bool
    answer: str | None
    sources: list[Source]
    mode: str
    confidence: float
    reason: str | None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the index and the model once, before the first request.
    app.state.service = AskService(load(), make_generator(settings))
    yield


app = FastAPI(title="AskDocs", version=__version__, lifespan=lifespan)


@app.get("/health")
def health(request: Request) -> dict:
    service: AskService = request.app.state.service
    return {"status": "ok", "version": __version__, "chunks": len(service.index.chunks), "mode": service.generator.mode}


@app.post("/ask", response_model=AskResponse)
def ask(body: AskRequest, request: Request) -> AskResponse:
    result = request.app.state.service.ask(body.question)
    return AskResponse(
        answered=result.answered,
        answer=result.answer,
        sources=[
            Source(ref=h.chunk.ref, title=h.chunk.title, url=h.chunk.url, excerpt=h.chunk.text[:300])
            for h in result.sources
        ],
        mode=result.mode,
        confidence=round(result.confidence, 3),
        reason=result.reason,
    )


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(STATIC / "index.html")
