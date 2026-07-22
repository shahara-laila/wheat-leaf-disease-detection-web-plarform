"""FastAPI application: wheat disease detection + treatment recommendation.

Implements Steps 6 (web integration) and 7 (privacy) of the paper.

Step 7 is enforced in exactly two places, both marked below:
  * /predict holds the upload in a local `bytes` and never writes it anywhere.
  * Logging records counts and class names only -- never query text, never bytes.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from . import config, knowledge
from .inference import Classifier, InvalidImage
from .schemas import (
    DiseasesResponse,
    HealthResponse,
    PredictResponse,
    RecommendRequest,
    RecommendResponse,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
log = logging.getLogger("wheat-api")

UPLOAD_CHUNK = 64 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the knowledge base and the model once, before serving traffic.

    Asymmetry is deliberate: a broken knowledge base is a broken deployment (it
    ships in the repo) and is fatal. A missing model is a setup step the user has
    yet to do, so the app degrades instead of dying.
    """
    app.state.db = knowledge.load_db(config.DB_PATH)
    app.state.index = knowledge.build_term_index(app.state.db)
    log.info(
        "knowledge base loaded: %d entries, %d terms",
        len(app.state.db), len(app.state.index),
    )

    app.state.classifier = Classifier(config.MODEL_PATH, config.CLASS_NAMES)
    app.state.classifier.load()
    yield
    app.state.classifier = None


app = FastAPI(
    title="Wheat Leaf Disease Detection API",
    description=(
        "CNN image classifier (Brown Rust / Healthy / Yellow Rust) plus a "
        "rule-based treatment recommender covering 13 wheat diseases."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# The bundled frontend is same-origin and needs no CORS. This exists only so a
# separately-served dev frontend can call the API. Note it is an explicit
# allowlist, not the wildcard the Flask version used.
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def no_cache_static(request: Request, call_next):
    """Force revalidation of the frontend assets.

    StaticFiles sends only ETag/Last-Modified, so browsers happily serve a stale
    styles.css or app.js from cache after an edit -- which looks exactly like a
    broken change. The payload is a few KB; correctness beats the caching here.
    """
    response = await call_next(request)
    if request.url.path.endswith((".css", ".js", ".html")) or request.url.path == "/":
        response.headers["Cache-Control"] = "no-cache, must-revalidate"
    return response


@app.exception_handler(HTTPException)
async def http_error(_: Request, exc: HTTPException) -> JSONResponse:
    detail = exc.detail
    if isinstance(detail, dict):
        return JSONResponse(status_code=exc.status_code, content=detail)
    return JSONResponse(status_code=exc.status_code, content={"error": detail})


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness probe. Returns 200 even when degraded, so it stays usable."""
    clf = app.state.classifier
    return HealthResponse(
        status="ok" if clf.ready else "degraded",
        model_loaded=clf.ready,
        model_error=clf.load_error,
        classes=config.CLASS_NAMES,
        diseases_known=sorted(app.state.db),
        input_size=list(clf.input_size) if clf.input_size else None,
        confidence_threshold=config.CONFIDENCE_THRESHOLD,
        kb_entries=len(app.state.db),
    )


@app.get("/diseases", response_model=DiseasesResponse)
async def diseases() -> DiseasesResponse:
    """Every disease in the knowledge base (13 diseases + Healthy)."""
    db = app.state.db
    return DiseasesResponse(
        count=len(db),
        diseases={
            name: {
                **entry,
                "detectable_by_image": name in config.CLASS_NAMES,
            }
            for name, entry in db.items()
        },
    )


async def _read_limited(upload: UploadFile) -> bytes:
    """Read an upload in chunks, aborting past the limit.

    Reading unbounded (`await upload.read()`) would let a large body exhaust
    memory before any size check could run.
    """
    chunks: list[bytes] = []
    total = 0
    while chunk := await upload.read(UPLOAD_CHUNK):
        total += len(chunk)
        if total > config.MAX_UPLOAD_BYTES:
            raise HTTPException(
                413,
                {
                    "error": "image too large",
                    "detail": f"limit is {config.MAX_UPLOAD_BYTES // (1024 * 1024)} MB",
                },
            )
        chunks.append(chunk)
    return b"".join(chunks)


@app.post("/predict", response_model=PredictResponse)
async def predict(image: UploadFile | None = File(None)) -> PredictResponse:
    """Classify a wheat leaf image.

    STEP 7 (privacy): the upload lives only in the local `raw` variable. It is
    never written to disk, never copied to a temp file, and goes out of scope
    when this function returns. Only the predicted class is logged.
    """
    if image is None or not image.filename:
        raise HTTPException(
            400, {"error": "send an image file under the 'image' field"}
        )

    clf = app.state.classifier
    if not clf.ready:
        raise HTTPException(
            503, {"error": "model unavailable", "detail": clf.load_error}
        )

    raw = await _read_limited(image)

    try:
        # Inference is blocking CPU work; off the event loop it goes.
        result = await run_in_threadpool(clf.predict, raw)
    except InvalidImage as exc:
        raise HTTPException(
            422, {"error": "could not decode image", "detail": str(exc)}
        ) from exc
    except Exception as exc:  # noqa: BLE001
        log.exception("inference failed")
        raise HTTPException(500, {"error": "inference failed"}) from exc

    # STEP 7: class and certainty only. No filename, no bytes, no dimensions.
    log.info("predict served: class=%s uncertain=%s", result.prediction, result.uncertain)

    return PredictResponse(
        prediction=result.prediction,
        confidence=result.confidence,
        probabilities=result.probabilities,
        recommendation=app.state.db[result.prediction]["recommendation"],
        uncertain=result.uncertain,
        margin=result.margin,
        message=result.message,
    )


@app.post("/recommend", response_model=RecommendResponse)
async def recommend(payload: RecommendRequest) -> RecommendResponse:
    """Map free-text symptoms or disease names to treatment recommendations.

    Independent of the CNN by design (paper Section 3.3), so this keeps working
    when the model is unavailable.

    STEP 7 (privacy): the query text is never logged -- only the match count.
    """
    result = knowledge.recommend_from_text(
        payload.text, app.state.db, app.state.index
    )
    # STEP 7: count only. Logging `payload.text` here would break the paper's
    # anonymization claim.
    log.info("recommend served: matches=%d", len(result["diseases"]))
    return RecommendResponse(**result)


# Mounted LAST so it cannot shadow the API routes above.
if config.FRONTEND_DIR.is_dir():
    app.mount(
        "/", StaticFiles(directory=config.FRONTEND_DIR, html=True), name="frontend"
    )
else:  # pragma: no cover
    log.warning("frontend directory missing at %s", config.FRONTEND_DIR)
