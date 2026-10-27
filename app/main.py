"""FastAPI entry point for VeriNews.

Run the server with:
    uvicorn app.main:app --host 127.0.0.1 --port 8000

The lifespan loads the spaCy, embedding, and NLI models once. The API always
returns a verdict object for a valid request; pipeline failures become
UNVERIFIED responses rather than HTTP 500 errors.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Protocol

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from starlette.concurrency import run_in_threadpool

import config
from app.pipeline import failure_verdict, get_pipeline
from app.schemas import Verdict

logger = logging.getLogger(__name__)


class VerifyRequest(BaseModel):
    claim: str = Field(
        ...,
        min_length=1,
        max_length=config.MAX_CLAIM_CHARS,
        description="A news claim to verify.",
    )

    @field_validator("claim")
    @classmethod
    def strip_claim(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Claim must not be empty.")
        return cleaned


class PipelineLike(Protocol):
    def warmup(self) -> None:
        ...

    def verify(self, claim_text: str) -> Verdict:
        ...


def create_app(
    *,
    pipeline: PipelineLike | None = None,
    warmup: bool = True,
) -> FastAPI:
    """Create the app. Tests pass a fake pipeline and disable warmup."""
    active_pipeline = pipeline or get_pipeline()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if warmup:
            logger.info("Loading VeriNews models at startup")
            await run_in_threadpool(active_pipeline.warmup)
        yield

    app = FastAPI(
        title="VeriNews",
        version="0.7.0",
        description="Evidence-based news claim verification.",
        lifespan=lifespan,
    )
    app.state.pipeline = active_pipeline

    @app.get("/health")
    def health() -> dict[str, object]:
        return {
            "status": "ok",
            "cache_only": config.CACHE_ONLY,
            "nli_model": config.NLI_MODEL_NAME,
        }

    @app.post("/verify")
    async def verify_claim(request: VerifyRequest) -> dict[str, object]:
        started = time.perf_counter()
        try:
            verdict = await run_in_threadpool(
                active_pipeline.verify,
                request.claim,
            )
        except Exception as exc:
            logger.exception("Unexpected pipeline failure")
            verdict = failure_verdict(
                request.claim,
                f"Unexpected pipeline failure ({type(exc).__name__})",
                latency_ms=round((time.perf_counter() - started) * 1000),
            )
        return asdict(verdict)

    if config.STATIC_DIR.is_dir():
        app.mount(
            "/",
            StaticFiles(directory=config.STATIC_DIR, html=True),
            name="static",
        )

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=False,
    )
