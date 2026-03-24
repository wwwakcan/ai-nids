"""
API endpoint tests (requires running server or TestClient)
Run: pytest tests/test_api.py -v
"""

import pytest
import numpy as np

# These tests use FastAPI TestClient — no running server needed
try:
    from fastapi.testclient import TestClient
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False


SAMPLE_FEATURES = list(np.random.RandomState(42).rand(41).round(4))

SAMPLE_PAYLOAD = {
    "src_ip":   "192.168.1.100",
    "dst_ip":   "10.0.0.1",
    "protocol": "tcp",
    "src_bytes": 1024,
    "dst_bytes": 512,
    "duration":  0.5,
    "features":  SAMPLE_FEATURES,
}


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not installed")
class TestAPIEndpoints:

    @pytest.fixture(scope="class")
    def client(self):
        import sys, os
        sys.path.insert(0, os.path.abspath("."))
        os.environ["SIEM_DRY_RUN"] = "true"
        from api.main import app
        return TestClient(app)

    def test_health_endpoint(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert data["status"] in ("ok", "degraded")

    def test_predict_returns_200_or_503(self, client):
        resp = client.post("/predict", json=SAMPLE_PAYLOAD)
        # 200 = models loaded, 503 = models not yet trained
        assert resp.status_code in (200, 503)

    def test_predict_response_schema(self, client):
        resp = client.post("/predict", json=SAMPLE_PAYLOAD)
        if resp.status_code == 200:
            data = resp.json()
            assert "label"      in data
            assert "severity"   in data
            assert "ae_score"   in data
            assert "confidence" in data
            assert data["label"] in ["Normal","DoS","Probe","R2L","U2R"]
            assert data["severity"] in ["CRITICAL","HIGH","MEDIUM","LOW","INFO"]
            assert 0.0 <= data["confidence"] <= 1.0

    def test_predict_missing_features(self, client):
        bad_payload = {**SAMPLE_PAYLOAD, "features": []}
        resp = client.post("/predict", json=bad_payload)
        assert resp.status_code == 422  # Validation error

    def test_metrics_endpoint(self, client):
        resp = client.get("/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_predictions" in data
        assert "attack_rate" in data


# ─── Standalone schema tests (no server needed) ───────────────────────────────

class TestSchemaValidation:

    def test_flow_request_valid(self):
        try:
            from api.schemas import FlowRequest
            req = FlowRequest(**SAMPLE_PAYLOAD)
            assert req.src_ip == "192.168.1.100"
            assert len(req.features) == 41
        except ImportError:
            pytest.skip("API schemas not available")

    def test_flow_request_rejects_empty_features(self):
        try:
            from api.schemas import FlowRequest
            from pydantic import ValidationError
            with pytest.raises(ValidationError):
                FlowRequest(**{**SAMPLE_PAYLOAD, "features": []})
        except ImportError:
            pytest.skip("API schemas not available")

    def test_prediction_response_has_timestamp(self):
        try:
            from api.schemas import PredictionResponse
            resp = PredictionResponse(
                label="DoS", severity="HIGH",
                ae_score=0.05, confidence=0.9, latency_ms=12.0
            )
            assert "Z" in resp.timestamp  # ISO timestamp ends with Z
        except ImportError:
            pytest.skip("API schemas not available")
