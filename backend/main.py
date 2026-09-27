"""
RetinaTriage - HTTP service.

Routes
    GET  /                      the interface
    GET  /api/health            model status and active thresholds
    GET  /api/config            labels, stage descriptions, threshold values
    GET  /api/metrics           evaluation results recorded during training
    GET  /api/samples           bundled example images
    POST /api/predict           grade an uploaded fundus photograph
    POST /api/predict/sample    grade one of the bundled examples
"""

from __future__ import annotations

import base64
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import demo as demo_mode
from .config import (
    GRADE_DESCRIPTIONS,
    GRADE_NAMES,
    GRADE_SHORT,
    PROJECT_ROOT,
    TRIAGE_ACTIONS,
    settings,
)
from .explain import class_activation_map, overlay_heatmap
from .inference import DRTriageEngine, ModelNotLoaded
from .metrics_store import MetricsStore
from .preprocessing import (
    looks_like_fundus,
    preprocess_fundus,
    preprocessing_stages,
    to_model_tensor,
)
from .schemas import AnalysisOut, HealthOut
from .triage import decide

VERSION = "1.2.0"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
)
logger = logging.getLogger("retinatriage")

FRONTEND_DIR = PROJECT_ROOT / "frontend"
SAMPLES_DIR = PROJECT_ROOT / "samples"

engine = DRTriageEngine(settings)
metrics = MetricsStore(settings)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup rather than on the first request."""
    started = time.perf_counter()
    if engine.load():
        logger.info(
            "Model ready in %.1fs (%s, %d-dim features)",
            time.perf_counter() - started,
            settings.backbone_file,
            engine.feature_size,
        )
    else:
        logger.warning("Model not loaded: %s", engine.load_error)
        if settings.allow_demo_mode:
            logger.warning("Serving in DEMO MODE - stage outputs are synthetic.")
    yield


app = FastAPI(
    title="RetinaTriage",
    description="Diabetic retinopathy stage detection and screening triage.",
    version=VERSION,
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def encode_image(image_bgr: np.ndarray, quality: int = 88) -> str:
    """BGR array -> base64 JPEG data URL, so results travel in one JSON body."""
    ok, buffer = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise HTTPException(500, "Could not encode the result image.")
    return "data:image/jpeg;base64," + base64.b64encode(buffer).decode("ascii")


def decode_upload(raw: bytes, filename: str) -> np.ndarray:
    """Upload bytes -> BGR array, with size and format checks."""
    limit = settings.max_upload_mb * 1024 * 1024
    if len(raw) > limit:
        raise HTTPException(
            413, f"That file is {len(raw) / 1e6:.1f} MB. The limit is {settings.max_upload_mb} MB."
        )
    if not raw:
        raise HTTPException(400, "The uploaded file is empty.")

    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(
            400,
            f"'{filename}' could not be read as an image. Upload a JPEG, PNG or TIFF "
            "fundus photograph.",
        )
    if min(image.shape[:2]) < 96:
        raise HTTPException(
            400, "That image is smaller than 96 pixels on one side. Upload the original capture."
        )
    return image


def analyse(image_bgr: np.ndarray, filename: str, want_stages: bool, want_heatmap: bool) -> dict:
    """The shared path behind both prediction endpoints."""
    request_id = uuid.uuid4().hex[:12]
    warnings: list[str] = []
    timing: dict[str, float] = {}

    if not looks_like_fundus(image_bgr):
        warnings.append(
            "This does not look like a retinal fundus photograph. The result below is "
            "very unlikely to be meaningful."
        )

    started = time.perf_counter()
    stages = preprocessing_stages(image_bgr, settings.image_size) if want_stages else None
    preprocessed = (
        stages["final"]
        if stages
        else preprocess_fundus(
            image_bgr, settings.image_size, settings.use_graham, settings.use_clahe
        )
    )
    timing["preprocess_ms"] = (time.perf_counter() - started) * 1000.0

    tensor = to_model_tensor(preprocessed, settings.image_size)

    using_demo = False
    heatmap_image = None

    if engine.is_ready:
        try:
            prediction = engine.predict(tensor)
        except ModelNotLoaded as error:  # pragma: no cover - defensive
            raise HTTPException(503, str(error)) from error

        if want_heatmap and engine.explains:
            started = time.perf_counter()
            heatmap = class_activation_map(engine, tensor)
            if heatmap is not None:
                heatmap_image = overlay_heatmap(preprocessed, heatmap)
            timing["explain_ms"] = (time.perf_counter() - started) * 1000.0
        elif want_heatmap:
            warnings.append(
                "Heatmaps need models/backbone_cam.onnx. Export it with "
                "scripts/export_deployment.py to turn this panel on."
            )
    elif settings.allow_demo_mode:
        using_demo = True
        prediction = demo_mode.synthetic_prediction(preprocessed, settings.mc_dropout_samples)
        warnings.append(
            "Demo mode: no trained weights are installed, so the stage shown is "
            "synthesised from image statistics, not predicted by the model."
        )
    else:
        raise HTTPException(503, engine.load_error or "The model is not available.")

    timing["inference_ms"] = prediction.inference_ms
    decision = decide(prediction, settings)

    images = {
        "original": encode_image(cv2.resize(image_bgr, (settings.image_size,) * 2)),
        "preprocessed": encode_image(preprocessed),
        "heatmap": encode_image(heatmap_image) if heatmap_image is not None else None,
    }
    if stages:
        images["stages"] = {
            name: encode_image(image, quality=80)
            for name, image in stages.items()
            if name != "final"
        }

    payload = prediction.to_dict()
    payload["grade_description"] = GRADE_DESCRIPTIONS[prediction.grade]
    payload["grade_probabilities"] = [
        {"grade": grade, "name": GRADE_SHORT[grade], "probability": round(probability, 5)}
        for grade, probability in enumerate(prediction.grade_probabilities)
    ]

    return {
        "request_id": request_id,
        "filename": filename,
        "demo_mode": using_demo,
        "warnings": warnings,
        "prediction": payload,
        "triage": decision.to_dict(),
        "images": images,
        "timing_ms": {key: round(value, 1) for key, value in timing.items()},
    }


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


@app.get("/api/health", response_model=HealthOut)
def health():
    return {
        "status": "ok" if engine.is_ready else ("demo" if settings.allow_demo_mode else "degraded"),
        "version": VERSION,
        "model": engine.describe(),
        "demo_mode": not engine.is_ready and settings.allow_demo_mode,
        "thresholds": {
            "ungradable": settings.ungradable_threshold,
            "uncertainty": settings.uncertainty_threshold,
            "confidence": settings.confidence_threshold,
            "referral_grade": settings.referral_grade,
        },
    }


@app.get("/api/config")
def configuration():
    return {
        "version": VERSION,
        "image_size": settings.image_size,
        "grades": [
            {
                "grade": grade,
                "name": GRADE_NAMES[grade],
                "short": GRADE_SHORT[grade],
                "description": GRADE_DESCRIPTIONS[grade],
            }
            for grade in range(5)
        ],
        "triage_actions": TRIAGE_ACTIONS,
        "thresholds": {
            "ungradable": settings.ungradable_threshold,
            "uncertainty": settings.uncertainty_threshold,
            "confidence": settings.confidence_threshold,
            "referral_grade": settings.referral_grade,
        },
        "mc_dropout_samples": settings.mc_dropout_samples,
    }


@app.get("/api/metrics")
def evaluation_metrics():
    data = metrics.everything()
    if not data["headline"].get("available"):
        return JSONResponse(
            {"available": False, "message": "artifacts/evaluation.json is missing."},
            status_code=200,
        )
    return data


@app.get("/api/samples")
def samples():
    if not SAMPLES_DIR.exists():
        return {"samples": []}
    files = sorted(
        path
        for path in SAMPLES_DIR.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    return {
        "samples": [
            {"name": path.name, "label": path.stem.replace("_", " "), "url": f"/samples/{path.name}"}
            for path in files
        ]
    }


@app.post("/api/predict", response_model=AnalysisOut)
async def predict(
    file: UploadFile = File(...),
    show_stages: bool = Form(True),
    show_heatmap: bool = Form(True),
):
    raw = await file.read()
    image = decode_upload(raw, file.filename or "upload")
    result = analyse(image, file.filename or "upload", show_stages, show_heatmap)
    logger.info(
        "%s  %s  grade=%d  action=%s  %.0fms",
        result["request_id"],
        result["filename"],
        result["prediction"]["grade"],
        result["triage"]["action"],
        sum(result["timing_ms"].values()),
    )
    return result


@app.post("/api/predict/sample", response_model=AnalysisOut)
def predict_sample(
    name: str = Form(...),
    show_stages: bool = Form(True),
    show_heatmap: bool = Form(True),
):
    # Resolve inside the samples folder only, so a crafted name cannot walk the
    # filesystem via ../.
    candidate = (SAMPLES_DIR / Path(name).name).resolve()
    if not str(candidate).startswith(str(SAMPLES_DIR.resolve())) or not candidate.exists():
        raise HTTPException(404, f"No bundled sample named '{name}'.")
    image = cv2.imread(str(candidate), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(500, f"Sample '{name}' could not be read.")
    return analyse(image, candidate.name, show_stages, show_heatmap)


# ---------------------------------------------------------------------------
# static files - mounted last so /api routes always win
# ---------------------------------------------------------------------------

if SAMPLES_DIR.exists():
    app.mount("/samples", StaticFiles(directory=SAMPLES_DIR), name="samples")

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(FRONTEND_DIR / "index.html")

    @app.get("/evidence", include_in_schema=False)
    def evidence():
        
        return FileResponse(FRONTEND_DIR / "evidence.html")
