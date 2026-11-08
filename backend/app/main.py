"""FastAPI application factory.

Run locally:   cd backend && uvicorn app.main:app --reload
Docs (auto):   http://localhost:8000/docs
"""

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import health, model, model_card, score
from app.config import Settings
from app.core.errors import ApiError, error_body
from app.core.predictor import ModelUnavailable, Predictor
from app.core.ratelimit import SlidingWindowLimiter

API_PREFIX = "/api/v1"
APP_VERSION = "1.0.0"

access_logger = logging.getLogger("heartguard.access")
if not access_logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(message)s"))
    access_logger.addHandler(_h)
    access_logger.setLevel(logging.INFO)
    access_logger.propagate = False


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    limiter = SlidingWindowLimiter(settings.rate_limit_per_minute)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Load once at startup -- never per request, never fit at serving time.
        try:
            app.state.predictor = Predictor.load(
                settings.model_path, settings.meta_path
            )
            access_logger.info(
                json.dumps(
                    {
                        "event": "model_loaded",
                        "model_version": app.state.predictor.version,
                    }
                )
            )
        except ModelUnavailable as exc:
            app.state.predictor = None
            access_logger.error(
                json.dumps({"event": "model_load_failed", "error": str(exc)})
            )
        yield

    app = FastAPI(
        title="HeartGuard AI - Diagnosis Support API",
        version=APP_VERSION,
        description="Educational clinical decision-support prototype. NOT a medical device.",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.predictor = None

    # ---- error handlers: one consistent body shape ----
    @app.exception_handler(ApiError)
    async def api_error_handler(request: Request, exc: ApiError):
        request.state.error_code = exc.error_code
        return JSONResponse(
            status_code=exc.status_code, content=error_body(exc.error_code, exc.message)
        )

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError):
        errs = exc.errors()
        if any(e.get("type") == "json_invalid" for e in errs):
            request.state.error_code = "MALFORMED_JSON"
            return JSONResponse(
                status_code=400,
                content=error_body("MALFORMED_JSON", "Request body is not valid JSON."),
            )
        fields = [
            {
                "field": ".".join(str(p) for p in list(e.get("loc", ()))[1:]) or "body",
                "message": str(e.get("msg", "Invalid value")),
            }
            for e in errs
        ]
        request.state.error_code = "VALIDATION_ERROR"
        return JSONResponse(
            status_code=422,
            content=error_body(
                "VALIDATION_ERROR", "One or more fields are invalid.", fields
            ),
        )

    @app.exception_handler(Exception)
    async def unexpected_handler(request: Request, exc: Exception):
        access_logger.error(
            json.dumps({"event": "unhandled_exception", "type": type(exc).__name__})
        )
        return JSONResponse(
            status_code=500,
            content=error_body("INTERNAL_ERROR", "Unexpected server error."),
        )

    # ---- access log + rate limit. NEVER logs request bodies or predictions (no PHI) ----
    @app.middleware("http")
    async def access_log(request: Request, call_next):
        request_id = uuid.uuid4().hex[:12]
        start = time.perf_counter()
        path = request.url.path
        limited = (
            path.startswith(API_PREFIX)
            and not path.endswith("/health")
            and not limiter.allow(request.client.host if request.client else "unknown")
        )
        if limited:
            request.state.error_code = "RATE_LIMITED"
            response = JSONResponse(
                status_code=429,
                headers={"Retry-After": "60"},
                content=error_body(
                    "RATE_LIMITED", "Too many requests. Please retry shortly."
                ),
            )
        else:
            response = await call_next(request)
        predictor = app.state.predictor
        access_logger.info(
            json.dumps(
                {
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "level": "INFO",
                    "request_id": request_id,
                    "endpoint": path,
                    "method": request.method,
                    "status_code": response.status_code,
                    "latency_ms": round((time.perf_counter() - start) * 1000, 2),
                    "model_version": predictor.version if predictor else None,
                    "error_code": getattr(request.state, "error_code", None),
                }
            )
        )
        response.headers["X-Request-ID"] = request_id
        return response

    # Added last => outermost, so even 429 responses carry CORS headers.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    for r in (health.router, score.router, model.router, model_card.router):
        app.include_router(r, prefix=API_PREFIX)
    return app


app = create_app()
