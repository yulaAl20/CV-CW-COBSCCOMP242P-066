"""
The downloadable report and the scan history page.

The PDF has to say exactly what the screen says, be built without touching the
disk, and survive the characters real file names contain. The history page has
to render with no scans and with several.
"""

from datetime import datetime
from pathlib import Path

import pytest

from backend.config import settings
from backend.inference import Prediction
from backend.report import build_report, report_filename
from backend.triage import decide

REPO_ROOT = Path(__file__).resolve().parent.parent


def _prediction(grade: int = 2, confidence: float = 0.8, ungradable: float = 0.05):
    probabilities = [0.05] * 5
    probabilities[grade] = confidence
    total = sum(probabilities)
    probabilities = [p / total for p in probabilities]
    return Prediction(
        grade=grade, grade_name=["No DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR",
                                 "Proliferative DR"][grade],
        expected_grade=float(grade), grade_probabilities=probabilities,
        confidence=max(probabilities), uncertainty=0.1, entropy=0.8,
        probability_ungradable=ungradable, probability_any_dr=0.9,
        probability_referable=0.7, mc_samples=20, inference_ms=180.0,
    )


def _jpeg() -> bytes:
    import cv2
    import numpy as np

    image = np.zeros((64, 64, 3), np.uint8)
    cv2.circle(image, (32, 32), 28, (30, 70, 160), -1)
    return cv2.imencode(".jpg", image)[1].tobytes()


def _build(prediction, **overrides) -> bytes:
    arguments = {
        "filename": "patient—0421 “left” eye….jpg",   # typographic characters on purpose
        "read_at": datetime(2026, 10, 1, 9, 30),
        "report_id": "0123456789abcdef",
        "prediction": prediction,
        "decision": decide(prediction, settings),
        "action_text": "Send this patient to an eye specialist.",
        "finding_text": "The model found moderate diabetic retinopathy.",
        "original_jpeg": _jpeg(),
        "second_jpeg": _jpeg(),
        "second_caption": "What the model sees",
        "warnings": [],
        "demo": False,
        "settings": settings,
    }
    arguments.update(overrides)
    return build_report(**arguments)


@pytest.mark.parametrize("grade, ungradable", [(0, 0.05), (2, 0.05), (4, 0.05), (2, 0.9)])
def test_report_is_a_single_page_pdf_for_every_kind_of_result(grade, ungradable):
    pdf = _build(_prediction(grade, ungradable=ungradable))
    assert pdf.startswith(b"%PDF-")
    assert b"/Count 1" in pdf, "the report should fit on one page"


def test_report_survives_warnings_demo_and_no_second_image():
    pdf = _build(_prediction(), warnings=["This does not look like a fundus photo."],
                 demo=True, second_jpeg=None)
    assert pdf.startswith(b"%PDF-")


def test_report_filename_is_safe_on_any_file_system():
    name = report_filename("IDRiD patient/0421 (left).jpg", datetime(2026, 10, 1, 9, 30))
    assert name == "retinatriage_IDRiD_patient_0421__left__20261001_0930.pdf"
    assert "/" not in name and " " not in name


def test_report_is_built_in_memory():
    """The model card promises photographs are stored nowhere on the server."""
    source = (REPO_ROOT / "backend" / "report.py").read_text()
    for banned in ("open(", "imwrite", ".output(\"", ".output('"):
        assert banned not in source, f"report.py writes to disk via {banned}"


def test_screen_and_report_share_their_wording():
    """The PDF must not be able to word a reading differently from the screen."""
    source = (REPO_ROOT / "streamlit_ui.py").read_text()
    report_part = source.split("def report_pdf")[1].split("\ndef ")[0]
    assert "action_sentence(decision)" in report_part
    assert "finding_sentence(prediction, decision, html=False)" in report_part


# -- the history page ---------------------------------------------------------

st_testing = pytest.importorskip("streamlit.testing.v1")


def test_history_page_renders_empty():
    app = st_testing.AppTest.from_file(str(REPO_ROOT / "views" / "history.py"),
                                       default_timeout=120)
    app.run()
    assert not app.exception, [e.value for e in app.exception]
    assert any("No scans yet" in info.value for info in app.info)


def test_history_page_lists_every_scan_with_view_and_pdf():
    from backend.triage import decide as decide_

    app = st_testing.AppTest.from_file(str(REPO_ROOT / "views" / "history.py"),
                                       default_timeout=120)
    entries = []
    for index, grade in enumerate((0, 3)):
        prediction = _prediction(grade)
        decision = decide_(prediction, settings)
        entries.append({
            "digest": f"digest{index}", "filename": f"scan{index}.jpg",
            "short_name": f"scan{index}.jpg", "read_at": datetime(2026, 10, 1, 9, index),
            "time": f"09:0{index}", "grade": grade, "grade_name": prediction.grade_name,
            "action_label": decision.label, "prediction": prediction,
            "decision": decision, "demo": False, "warnings": [],
            "original": _jpeg(), "preprocessed": _jpeg(), "heatmap": None,
            "stages": {}, "timing": {},
        })
    app.session_state["history"] = entries
    app.run()
    assert not app.exception, [e.value for e in app.exception]

    views = [button for button in app.button if button.label == "View"]
    assert len(views) == 2
    downloads = [d.proto.label for d in app.get("download_button")]
    assert downloads.count("PDF") == 2 and "Download all (CSV)" in downloads
    assert all(entry.get("report_pdf", b"").startswith(b"%PDF-")
               for entry in app.session_state["history"])
