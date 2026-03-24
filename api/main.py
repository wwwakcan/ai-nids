"""
AI-NIDS FastAPI Inference Endpoint
===================================
POST /predict  — classify a single network flow
GET  /health   — service health check
GET  /metrics  — model performance counters
"""

from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import pickle, os, time, logging
import numpy as np

from api.schemas import FlowRequest, PredictionResponse, HealthResponse
from api.siem_client import send_to_elasticsearch, trigger_webhook

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ai-nids")

# ── Globals ───────────────────────────────────────────────────────────────────
MODEL_DIR  = os.getenv("MODEL_DIR", "models")
ensemble   = None
scaler     = None
iforest    = None
autoencoder = None
COUNTERS   = {"total": 0, "attacks": 0, "anomalies": 0, "errors": 0}

CLASS_NAMES = {0: "Normal", 1: "DoS", 2: "Probe", 3: "R2L", 4: "U2R"}

SEVERITY_MAP = {
    "U2R":    "CRITICAL",
    "R2L":    "CRITICAL",
    "DoS":    "HIGH",
    "Probe":  "MEDIUM",
    "Normal": "INFO",
}

# ── Startup / shutdown ────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global ensemble, scaler, iforest, autoencoder
    logger.info("Loading models...")

    try:
        with open(f"{MODEL_DIR}/ensemble_rf_svm.pkl", "rb") as f:
            ensemble = pickle.load(f)
        with open(f"{MODEL_DIR}/scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        with open(f"{MODEL_DIR}/isolation_forest.pkl", "rb") as f:
            iforest = pickle.load(f)
        logger.info("Models loaded successfully")
    except FileNotFoundError:
        logger.warning("Model files not found — run src/train.py first")

    try:
        import tensorflow as tf
        autoencoder = tf.keras.models.load_model(f"{MODEL_DIR}/lstm_autoencoder.keras")
        logger.info("LSTM Autoencoder loaded")
    except Exception as e:
        logger.warning(f"Autoencoder not loaded: {e}")

    yield
    logger.info("Shutting down AI-NIDS API")


app = FastAPI(
    title="AI-NIDS Inference API",
    description="AI-Powered Network Intrusion Detection System",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Helpers ───────────────────────────────────────────────────────────────────
def classify_severity(label: str, ae_score: float, tau: float = 0.05) -> str:
    if ae_score > tau * 3:
        return "CRITICAL"
    return SEVERITY_MAP.get(label, "MEDIUM")


def get_ae_score(features: np.ndarray) -> float:
    """Compute LSTM autoencoder reconstruction error."""
    if autoencoder is None:
        return 0.0
    x = features.reshape(1, 1, -1).astype("float32")
    recon = autoencoder.predict(x, verbose=0)
    return float(np.mean(np.power(x - recon, 2)))


# ── Routes ────────────────────────────────────────────────────────────────────
@app.post("/predict", response_model=PredictionResponse)
async def predict(req: FlowRequest):
    COUNTERS["total"] += 1
    t0 = time.time()

    if ensemble is None or scaler is None:
        COUNTERS["errors"] += 1
        raise HTTPException(503, "Models not loaded. Run src/train.py first.")

    try:
        features = np.array(req.features, dtype=float).reshape(1, -1)
        features_scaled = scaler.transform(features)

        # M1 — supervised classification
        label_idx  = int(ensemble.predict(features_scaled)[0])
        label      = CLASS_NAMES.get(label_idx, "Unknown")
        proba      = float(ensemble.predict_proba(features_scaled)[0].max())

        # M2 — anomaly score
        ae_score   = get_ae_score(features_scaled)
        if_score   = float(-iforest.decision_function(features_scaled)[0]) if iforest else 0.0

        # Severity
        severity   = classify_severity(label, ae_score)
        latency_ms = round((time.time() - t0) * 1000, 2)

        if label != "Normal":
            COUNTERS["attacks"] += 1
        if ae_score > 0.05:
            COUNTERS["anomalies"] += 1

        event = {
            "label":      label,
            "severity":   severity,
            "ae_score":   round(ae_score, 6),
            "if_score":   round(if_score, 6),
            "confidence": round(proba, 4),
            "src_ip":     req.src_ip,
            "dst_ip":     req.dst_ip,
            "protocol":   req.protocol,
            "latency_ms": latency_ms,
        }

        # Forward to SIEM
        send_to_elasticsearch(event)
        if severity in ("CRITICAL", "HIGH"):
            trigger_webhook(severity, event)

        logger.info(f"[{severity}] {label} from {req.src_ip} ({latency_ms}ms)")

        return PredictionResponse(
            label=label,
            severity=severity,
            ae_score=round(ae_score, 6),
            confidence=round(proba, 4),
            latency_ms=latency_ms,
        )

    except Exception as e:
        COUNTERS["errors"] += 1
        logger.error(f"Prediction error: {e}")
        raise HTTPException(500, str(e))


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok" if ensemble is not None else "degraded",
        models_loaded=ensemble is not None,
        autoencoder_loaded=autoencoder is not None,
        total_predictions=COUNTERS["total"],
    )


@app.get("/metrics")
async def metrics():
    return {
        "total_predictions": COUNTERS["total"],
        "total_attacks":     COUNTERS["attacks"],
        "total_anomalies":   COUNTERS["anomalies"],
        "total_errors":      COUNTERS["errors"],
        "attack_rate":       round(COUNTERS["attacks"] / max(COUNTERS["total"], 1), 4),
    }
