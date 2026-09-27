
from fastapi.testclient import TestClient

from backend.main import app


def test_reading_page_and_evidence_page_are_served():
    
    with TestClient(app) as client:
        reading = client.get("/")
        evidence = client.get("/evidence")

        assert reading.status_code == 200
        assert evidence.status_code == 200

        assert 'id="evidence-cta"' in reading.text, "no link through to the evidence"
        assert 'id="performance"' not in reading.text, "the old panel is still here"
        assert 'id="confusion"' not in reading.text, "charts are still on the reading page"

        assert 'id="confusion"' in evidence.text
        assert 'id="comparison"' in evidence.text


def test_static_assets_are_reachable():
    with TestClient(app) as client:
        assert client.get("/static/css/styles.css").status_code == 200
        for script in ("grades.js", "api.js", "charts.js", "app.js", "evidence.js"):
            assert client.get(f"/static/js/{script}").status_code == 200, script


def test_metrics_endpoint_carries_what_the_evidence_page_needs():
    with TestClient(app) as client:
        data = client.get("/api/metrics").json()

    for key in ("headline", "per_class", "confusion", "history",
                "referral_sweep", "comparison", "run", "deployment"):
        assert key in data, f"/api/metrics is missing '{key}'"

    # the external comparison is the reason the page exists, so check it computes
    accuracy = next(row for row in data["comparison"] if row["metric"] == "Stage accuracy")
    assert accuracy["gap"] == round(accuracy["internal"] - accuracy["external"], 4)


def test_health_reports_whether_the_model_loaded():
    with TestClient(app) as client:
        health = client.get("/api/health").json()

    assert health["status"] in {"ok", "demo", "degraded"}
    assert "ungradable" in health["thresholds"]
