"""
Streamlit front-end tests.

`AppTest` runs the page script in-process and exposes the elements it produced,
so these check that the pages actually execute and render the right numbers —
not merely that the server returns a 200 for the app shell, which it does even
when the script raises.
"""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("streamlit", reason="streamlit is optional for the FastAPI service")

from streamlit.testing.v1 import AppTest  # noqa: E402

# AppTest resolves a relative path against the file that calls it, which would
# look inside tests/. Anchor everything to the repository root instead.
REPO_ROOT = Path(__file__).resolve().parent.parent


def _sample_fundus() -> bytes:
    """A synthetic fundus-like image: bright disc on a dark field, with lesions."""
    import cv2
    import numpy as np

    image = np.zeros((700, 700, 3), np.uint8)
    cv2.circle(image, (350, 350), 300, (28, 60, 150), -1)
    rng = np.random.default_rng(3)
    for _ in range(400):
        x, y = rng.integers(120, 580, 2)
        if (x - 350) ** 2 + (y - 350) ** 2 < 290 ** 2:
            cv2.circle(image, (int(x), int(y)), int(rng.integers(2, 7)), (20, 30, 200), -1)
    cv2.circle(image, (460, 330), 52, (60, 160, 235), -1)
    return cv2.imencode(".jpg", cv2.GaussianBlur(image, (0, 0), 3))[1].tobytes()


def run(path: str, timeout: int = 120, upload: bool = False) -> AppTest:
    app = AppTest.from_file(str(REPO_ROOT / path), default_timeout=timeout)
    app.run()
    assert not app.exception, f"{path} raised: {[e.value for e in app.exception]}"

    if upload:
        app.get("file_uploader")[0].upload("fundus.jpg", _sample_fundus(), "image/jpeg").run()
        assert not app.exception, f"{path} raised on upload: {[e.value for e in app.exception]}"
    return app


def text_of(app: AppTest) -> str:
    """Everything the page rendered, flattened, for substring assertions."""
    parts = []
    for collection in (app.markdown, app.title, app.header, app.subheader,
                       app.caption, app.warning, app.error, app.info, app.success):
        parts += [element.value for element in collection]
    return "\n".join(str(part) for part in parts)


# -- the reading page --------------------------------------------------------


def test_reading_page_runs_without_an_upload():
    """The landing state must render; it is what a reviewer sees first."""
    app = run("views/reader.py")
    assert "Read a fundus photograph" in text_of(app)
    assert app.file_uploader, "no upload control on the reading page"


def test_reading_page_still_exposes_the_thresholds():
    """
    They moved into an expander to declutter the sidebar, but they must remain
    reachable — a reader who cannot see what the app is configured to do cannot
    judge the reading.
    """
    app = run("views/reader.py")
    labels = [expander.label for expander in app.expander]
    assert "Decision thresholds" in labels
    assert "Refer at stage" in text_of(app)


def test_the_numbers_start_hidden_but_are_present():
    """
    Progressive disclosure, not omission: the working is one click away, never
    thrown away. A reading nobody can check is worse than no reading.
    """
    app = run("views/reader.py", upload=True)
    reveals = [e for e in app.expander if "numbers behind this reading" in e.label]
    assert reveals, "no reveal for the technical detail"
    assert not reveals[0].proto.expanded, "the numbers should start collapsed"

    body = text_of(app)
    assert "Entropy" in body, "entropy is missing from the reveal"
    assert "Continuous severity" in body
    assert "Rule" in body


def test_the_answer_comes_before_the_evidence():
    """The action must be stated in words a screener can act on, up front."""
    body = text_of(run("views/reader.py", upload=True))
    plain = ("Send this patient to an eye specialist",
             "This one needs a person to look at it",
             "No referral needed",
             "This photograph is not clear enough to read")
    assert any(sentence in body for sentence in plain), \
        "no plain-language action on the page"
    assert "In short" in body, "the plain yes/no answers are missing"


def test_a_reading_is_kept_in_the_history():
    app = run("views/reader.py", upload=True)
    history = app.session_state["history"]
    assert len(history) == 1
    entry = history[0]
    assert entry["filename"].endswith(".jpg")
    assert 0 <= entry["grade"] <= 4
    assert entry["action_label"]
    # stored as JPEG bytes, not arrays: ten raw 384x384 entries would be ~40 MB
    assert isinstance(entry["original"], bytes)
    assert len(entry["original"]) < 200_000


def test_the_same_photograph_is_not_graded_twice():
    """Re-running must not re-infer; that is what the digest guard is for."""
    app = run("views/reader.py", upload=True)
    assert len(app.session_state["history"]) == 1
    app.run()
    assert len(app.session_state["history"]) == 1, "re-ran inference on a rerun"


def test_history_is_capped_so_a_long_session_cannot_grow_without_bound():
    import streamlit_ui
    assert streamlit_ui.HISTORY_LIMIT <= 12


def test_nothing_is_written_to_disk():
    """
    The model card promises retinal photographs are stored nowhere — they are
    biometric identifiers. History lives in session state only.
    """
    import streamlit_ui

    source = (REPO_ROOT / "streamlit_ui.py").read_text()
    assert "session_state" in source
    for banned in ("open(", "to_csv", "imwrite", "pickle.dump"):
        assert banned not in source.split("def _fetch_weights")[0], \
            f"streamlit_ui.py writes to disk via {banned}"
    assert streamlit_ui.HISTORY_LIMIT


def test_reading_page_links_to_the_evidence():
    """
    A readout that cannot be checked is worse than no readout, and the link
    must be there BEFORE an upload — the script stops early without a file, so
    a visitor with nothing to hand would otherwise never see it.
    """
    # st.navigation declares both pages up front, so the evidence page is in
    # the sidebar from the first paint rather than only after a result.
    source = (REPO_ROOT / "streamlit_app.py").read_text()
    assert "views/evidence.py" in source
    assert "Model evidence" in source


def test_reading_page_carries_the_disclaimer():
    assert "Not a medical device" in text_of(run("views/reader.py"))


# -- the evidence page -------------------------------------------------------


def test_evidence_page_renders_the_recorded_results():
    body = text_of(run("views/evidence.py"))
    assert "What this model gets right" in body
    # read from artifacts/evaluation.json, not hard-coded in the page
    assert "86.6%" in body, "the recorded DDR accuracy is not on the page"
    assert "0.907" in body, "the recorded QWK is not on the page"


def test_evidence_page_states_its_limitations():
    """
    The quality-gate gap is the most serious limitation in the project. It is
    stated in the product, not only in the report, and this pins that.
    """
    body = text_of(run("views/evidence.py"))
    assert "never seen a bad photograph" in body
    assert "clinical validation" in body.lower()


def test_evidence_page_shows_the_generalisation_gap():
    app = run("views/evidence.py")
    frames = [frame.value for frame in app.dataframe]
    assert frames, "no tables rendered on the evidence page"

    comparison = next(
        (f for f in frames if hasattr(f, "columns") and "APTOS 2019" in list(f.columns)),
        None,
    )
    assert comparison is not None, "the internal/external comparison table is missing"

    row = comparison[comparison["Metric"] == "Referable DR AUC"].iloc[0]
    # the finding: ranking survives the move to new data, staging does not
    assert row["Gap"] < 0.05, "referable AUC should barely move on external data"


# -- the shared layer --------------------------------------------------------


def test_both_interfaces_share_one_severity_palette():
    """A stage must mean the same colour in both front ends."""
    import re
    import sys

    sys.path.insert(0, str(REPO_ROOT))
    import streamlit_ui

    source = (REPO_ROOT / "frontend/js/grades.js").read_text()
    in_js = re.findall(r'"(#[0-9A-Fa-f]{6})"', source)[:5]
    assert [c.upper() for c in in_js] == [c.upper() for c in streamlit_ui.GRADE_COLOURS]


def test_streamlit_layer_holds_no_decision_logic():
    """
    The Streamlit files are presentation only. If grading or triage logic ever
    gets written here it can drift from the FastAPI service and from the
    notebook, and the evaluation stops describing the app.
    """
    banned = ("def corn_", "def decide(", "def preprocess_fundus", "cumprod")
    for path in ("streamlit_app.py", "streamlit_ui.py", "views/evidence.py"):
        source = (REPO_ROOT / path).read_text()
        for token in banned:
            assert token not in source, f"{path} reimplements '{token}' — import it instead"


def test_cam_can_be_switched_off_for_a_small_host():
    """
    The heatmap needs a second ONNX session. On a 1 GB host that is the
    difference between running and being OOM-killed, so it must be optional.
    """
    from backend.config import Settings
    from backend.inference import DRTriageEngine

    settings = Settings()
    settings.enable_cam = False
    engine = DRTriageEngine(settings)
    engine.load()
    assert engine.is_ready, engine.load_error
    assert not engine.explains, "CAM loaded despite enable_cam=False"

    prediction = engine.predict(
        np.zeros((1, 3, settings.image_size, settings.image_size), dtype=np.float32)
    )
    assert 0 <= prediction.grade <= 4


def test_stand_in_weights_are_called_out():
    """
    The stand-in loads cleanly and grades every image the same way. If the app
    reported "Model ready" for it, a screenshot of a meaningless reading would
    look exactly like a real one — which is how this went unnoticed once.
    """
    from backend.config import Settings
    from backend.inference import DRTriageEngine

    engine = DRTriageEngine(Settings())
    engine.load()
    assert engine.is_ready
    assert engine.is_stand_in, "the testing weights were not recognised as stand-in"

    body = text_of(run("views/reader.py"))
    assert "stand-in weights" in body.lower()


def test_a_reading_from_stand_in_weights_is_marked_synthetic():
    app = run("views/reader.py", upload=True)
    entry = app.session_state["history"][0]
    assert entry["demo"], "a stand-in reading was not flagged as synthetic"
