
from __future__ import annotations

import base64
import os
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from backend.config import GRADE_DESCRIPTIONS, settings
from backend.inference import DRTriageEngine
from backend.metrics_store import MetricsStore

# The severity scale is the whole colour system. Defined once here, matching
# frontend/js/grades.js, so a stage means the same colour in both interfaces.
GRADE_COLOURS = ["#3FB98A", "#BFC94E", "#E9A13B", "#E2663C", "#CF3D57"]
GRADE_SHORT = ["No DR", "Mild", "Moderate", "Severe", "PDR"]

URGENCY_COLOURS = {
    "routine": "#3FB98A",
    "soon": "#E9A13B",
    "urgent": "#CF3D57",
    "blocked": "#8298AE",
}

PROJECT_ROOT = Path(__file__).resolve().parent

THEME_CSS = """
<style>
  /* Fundus photographs are read on dark screens so the reds of the retina stay
     true. The interface follows, and the five severity hues appear nowhere
     else, so a colour on screen always means a stage. */
  .stApp { background: #0E1620; }

  html, body, [class*="css"] { font-family: "IBM Plex Sans", system-ui, sans-serif; }

  .block-container { padding-top: 2.4rem; max-width: 1180px; }

  h1, h2, h3 { letter-spacing: -0.015em; }

  /* measurements sit in columns, so they are monospaced and tabular */
  .num { font-family: "IBM Plex Mono", ui-monospace, monospace;
         font-variant-numeric: tabular-nums; }

  .dr-panel {
    background: #16212E; border: 1px solid #1E2C3D;
    border-radius: 8px; padding: 20px;
  }

  .dr-eyebrow { font-size: 12.5px; color: #6B7F95; margin: 0; }
  .dr-note    { font-size: 12.5px; color: #6B7F95; margin: 4px 0 0; }
  .dr-lede    { font-size: 15px; color: #97A9BD; max-width: 68ch; }

  /* ── severity ladder ─────────────────────────────────────────────── */
  .ladder-wrap { position: relative; margin: 26px 0 22px; }
  .ladder-track { display: grid; grid-template-columns: repeat(5, 1fr); gap: 3px; }
  .ladder-step {
    height: 46px; border-radius: 2px; display: flex;
    align-items: flex-end; justify-content: center; padding-bottom: 6px;
    font-size: 11.5px; color: #6B7F95; background: #1C2A3A;
    border-bottom: 2px solid #26364A;
  }
  .ladder-step.on { color: #0E1620; font-weight: 600; }
  .ladder-band {
    position: absolute; top: 0; height: 46px;
    background: rgba(221,231,241,.13);
    border-left: 1px solid rgba(221,231,241,.3);
    border-right: 1px solid rgba(221,231,241,.3);
  }
  .ladder-marker { position: absolute; top: -8px; transform: translateX(-50%); }
  .ladder-marker-label {
    font-family: "IBM Plex Mono", monospace; font-size: 11px; font-weight: 600;
    background: #DDE7F1; color: #0E1620; padding: 1px 6px; border-radius: 2px;
    white-space: nowrap;
  }
  .ladder-marker-line { width: 2px; height: 54px; background: #DDE7F1; margin: 0 auto; }

  /* ── stage bars ──────────────────────────────────────────────────── */
  .bar-row { display: grid; grid-template-columns: 82px 1fr 52px;
             align-items: center; gap: 10px; margin-bottom: 9px; }
  .bar-name { font-size: 12.5px; color: #97A9BD; }
  .bar-track { height: 8px; background: #1C2A3A; border-radius: 2px; overflow: hidden; }
  .bar-fill { height: 100%; border-radius: 2px; }
  .bar-val { font-family: "IBM Plex Mono", monospace; font-variant-numeric: tabular-nums;
             font-size: 12px; text-align: right; color: #97A9BD; }
  .bar-row.top .bar-name, .bar-row.top .bar-val { color: #DDE7F1; font-weight: 500; }

  /* ── triage card ─────────────────────────────────────────────────── */
  .triage { border-left: 3px solid #26364A; }
  .triage-label  { font-size: 19px; font-weight: 600; margin: 3px 0 0; }
  .triage-reason { font-size: 13.5px; color: #97A9BD; margin-top: 8px; }
  .triage-rule   { font-size: 12px; color: #6B7F95; margin-top: 10px; }

  /* ── metric tiles ────────────────────────────────────────────────── */
  .tile { background: #16212E; border: 1px solid #1E2C3D; border-radius: 8px;
          padding: 16px 18px; height: 100%; }
  .tile-val { font-family: "IBM Plex Mono", monospace; font-variant-numeric: tabular-nums;
              font-size: 24px; font-weight: 500; letter-spacing: -0.02em; color: #DDE7F1; }
  .tile-lab { font-size: 12.5px; color: #97A9BD; margin-top: 4px; }
  .tile-sub { font-size: 11.5px; color: #6B7F95; margin-top: 2px; }

  .disclaimer { border-top: 1px solid #1E2C3D; margin-top: 48px; padding-top: 22px;
                font-size: 13px; color: #6B7F95; }
  .disclaimer strong { color: #CF3D57; }

  [data-testid="stSidebarNav"] { padding-top: 12px; }
</style>
"""


def apply_theme() -> None:
    st.markdown(THEME_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# model loading
# ---------------------------------------------------------------------------


def _fetch_weights(base_url: str, target: Path) -> list[str]:
    """Pull the weights from a GitHub Release on first run."""
    target.mkdir(parents=True, exist_ok=True)
    wanted = ["backbone_fp32.onnx", "heads.npz"]
    if settings.enable_cam:
        wanted.append("backbone_cam.onnx")

    problems = []
    for name in wanted:
        destination = target / name
        if destination.exists():
            continue
        try:
            with st.spinner(f"Fetching {name} (first run only)…"):
                urllib.request.urlretrieve(f"{base_url.rstrip('/')}/{name}", destination)
        except Exception as error:
            destination.unlink(missing_ok=True)
            problems.append(f"{name}: {error}")
    return problems


def _weights_url() -> str:
    """
    Where to fetch the weights from, if they are not already on disk.

    `st.secrets` RAISES when no secrets.toml exists rather than returning a
    default, which is the normal state on a fresh clone and on every local run.
    So the lookup is guarded: the environment variable wins, secrets are a
    fallback for Streamlit Community Cloud, and neither being present is fine.
    """
    from_env = os.environ.get("DR_WEIGHTS_URL", "")
    if from_env:
        return from_env
    try:
        return str(st.secrets.get("DR_WEIGHTS_URL", ""))
    except Exception:
        return ""


@st.cache_resource(show_spinner="Loading the model…")
def load_engine() -> DRTriageEngine:
    """
    Load the ONNX session once per server process.

    `cache_resource` rather than `cache_data` because the session is a live
    object holding ~80 MB of weights, not a value to copy. Without this the
    model would reload on every widget interaction and each click would cost
    several seconds.
    """
    weights_url = _weights_url()
    if weights_url and not settings.backbone_path.exists():
        _fetch_weights(weights_url, settings.model_dir)

    engine = DRTriageEngine(settings)
    engine.load()
    return engine


@st.cache_data(show_spinner=False)
def load_metrics() -> dict:
    """The recorded training results. Small, immutable, so cache the value."""
    return MetricsStore(settings).everything()


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------


def rgb(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def png_data_uri(image_bgr: np.ndarray) -> str:
    ok, buffer = cv2.imencode(".png", image_bgr)
    return "data:image/png;base64," + base64.b64encode(buffer).decode() if ok else ""


def severity_ladder(prediction) -> None:
    """
    The signature readout: the continuous severity score as a position on the
    five-stage scale, with the spread of the repeated passes as a band.

    A discrete label would hide the thing that matters most about a borderline
    case — that it is borderline. 1.9 and 2.4 both round to stage 2 and mean
    quite different things.
    """
    steps = []
    for grade, colour in enumerate(GRADE_COLOURS):
        if grade == prediction.grade:
            style = f"background:{colour};border-bottom-color:{colour}"
            steps.append(f'<div class="ladder-step on" style="{style}">{GRADE_SHORT[grade]}</div>')
        else:
            tint = f"border-bottom-color:color-mix(in srgb,{colour} 45%,transparent)"
            steps.append(f'<div class="ladder-step" style="{tint}">{GRADE_SHORT[grade]}</div>')

    position = (prediction.expected_grade + 0.5) / 5
    spread = prediction.uncertainty / 5

    band = ""
    if spread > 0.002:
        left = max(0.0, position - spread)
        width = min(1.0, position + spread) - left
        band = f'<div class="ladder-band" style="left:{left * 100:.1f}%;width:{width * 100:.1f}%"></div>'

    marker_left = min(max(position, 0.02), 0.98) * 100
    marker = (
        f'<div class="ladder-marker" style="left:{marker_left:.1f}%">'
        f'<div class="ladder-marker-label">{prediction.expected_grade:.2f}</div>'
        f'<div class="ladder-marker-line"></div></div>'
    )

    st.markdown(
        f'<div class="ladder-wrap"><div class="ladder-track">{"".join(steps)}</div>'
        f"{band}{marker}</div>",
        unsafe_allow_html=True,
    )

    legend = (
        f"The marker is the continuous severity score. The shaded band is how far "
        f"{prediction.mc_samples} repeated passes disagreed with each other."
        if spread > 0.002
        else "The marker is the continuous severity score. Repeated passes agreed closely."
    )
    st.markdown(f'<p class="dr-note">{legend}</p>', unsafe_allow_html=True)


def stage_bars(prediction, heading: str = "How the five stages scored") -> None:
    """
    Emitted as ONE markdown call, panel included.

    Streamlit sanitises each `st.markdown` separately, so an opening <div> in
    one call and its </div> in another do not pair up — they render as an empty
    box followed by unwrapped content.
    """
    rows = []
    for grade, probability in enumerate(prediction.grade_probabilities):
        top = " top" if grade == prediction.grade else ""
        opacity = 1.0 if grade == prediction.grade else 0.42
        width = max(probability * 100, 0.6)
        rows.append(
            f'<div class="bar-row{top}">'
            f'<span class="bar-name">{GRADE_SHORT[grade]}</span>'
            f'<div class="bar-track"><div class="bar-fill" style="width:{width:.1f}%;'
            f'background:{GRADE_COLOURS[grade]};opacity:{opacity}"></div></div>'
            f'<span class="bar-val">{probability * 100:.1f}%</span></div>'
        )
    st.markdown(
        f'<div class="dr-panel"><p style="font-weight:600;font-size:14.5px;'
        f'margin:0 0 14px">{heading}</p>{"".join(rows)}</div>',
        unsafe_allow_html=True,
    )


def triage_card(decision) -> None:
    colour = URGENCY_COLOURS.get(decision.urgency, "#26364A")
    st.markdown(
        f'<div class="dr-panel triage" style="border-left-color:{colour}">'
        f'<p class="dr-eyebrow">Next step</p>'
        f'<p class="triage-label">{decision.label}</p>'
        f'<p class="triage-reason">{decision.reason}</p>'
        f'<p class="triage-rule">Decided by rule {decision.rule} of 4 — '
        f"quality, then certainty, then severity.</p></div>",
        unsafe_allow_html=True,
    )


def tile(value: str, label: str, sub: str = "") -> None:
    st.markdown(
        f'<div class="tile"><p class="tile-val">{value}</p>'
        f'<p class="tile-lab">{label}</p>'
        + (f'<p class="tile-sub">{sub}</p>' if sub else "")
        + "</div>",
        unsafe_allow_html=True,
    )


def grade_description(grade: int) -> str:
    return GRADE_DESCRIPTIONS[grade]


def disclaimer() -> None:
    st.markdown(
        '<div class="disclaimer"><strong>Not a medical device.</strong> '
        "RetinaTriage is a student prototype built for coursework. It has not been "
        "clinically validated and must not be used to make decisions about a real "
        "patient.</div>",
        unsafe_allow_html=True,
    )


def model_status(engine: DRTriageEngine) -> None:
    """One line in the sidebar saying whether the numbers are real."""
    if engine.is_ready and engine.is_stand_in:
        st.sidebar.error("Stand-in weights — results are meaningless")
    elif engine.is_ready:
        size = engine.describe().get("backbone_size_mb")
        #st.sidebar.success(f"Model ready{f' · {size} MB' if size else ''}")
        if not engine.explains:
            #st.sidebar.caption("Heatmaps off — backbone_cam.onnx not loaded.")
    elif not settings.allow_demo_mode:
        st.sidebar.error("Model unavailable")


# ---------------------------------------------------------------------------
# plain language
# ---------------------------------------------------------------------------
# The person reading a fundus photograph in a screening clinic wants to know
# what to do with the patient. The probability vector is the evidence for that
# decision, not the decision — so it goes behind a reveal, and the answer comes
# first in words anyone can act on.

PLAIN_FINDING = {
    0: "No signs of diabetic retinopathy",
    1: "Earliest signs of diabetic retinopathy",
    2: "Moderate diabetic retinopathy",
    3: "Severe diabetic retinopathy",
    4: "Advanced, sight-threatening diabetic retinopathy",
}

PLAIN_ACTION = {
    "recapture": "This photograph is not clear enough to read. Take another one.",
    "human": "This one needs a person to look at it.",
    "ophthalmology": "Send this patient to an eye specialist.",
    "routine": "No referral needed. Screen again at the usual interval.",
}

STAGE_DOT = ["🟢", "🟡", "🟠", "🔴", "🔴"]


def confidence_phrase(prediction) -> str:
    """How sure the model is, in words rather than a percentage."""
    confidence = prediction.confidence
    if confidence >= 0.85:
        return "and is confident about it"
    if confidence >= 0.70:
        return "and is fairly confident about it"
    if confidence >= 0.55:
        return "but is not very confident"
    return "but is unsure"


def plain_summary(prediction, decision, demo: bool = False) -> None:
    """The answer, before any of the evidence for it."""
    colour = URGENCY_COLOURS.get(decision.urgency, "#26364A")
    finding = PLAIN_FINDING[prediction.grade]
    action = PLAIN_ACTION.get(decision.action, decision.label)

    if decision.action == "recapture":
        # the stage is not trustworthy here, so do not lead with it
        detail = "The image quality check rejected this photograph before grading it."
    elif decision.action == "human":
        detail = (
            f"The model read it as <strong>{finding.lower()}</strong>, "
            f"{confidence_phrase(prediction)}. That is not certain enough to report "
            "on its own."
        )
    else:
        detail = (
            f"The model found <strong>{finding.lower()}</strong>, "
            f"{confidence_phrase(prediction)}."
        )

    scale = (
        ""
        if decision.action == "recapture"
        else f'<p style="font-size:13px;color:#6B7F95;margin-top:14px">'
        f"Stage {prediction.grade} of 4 on the international diabetic "
        f"retinopathy scale.</p>"
    )

    flag = (
        '<p style="font-size:12.5px;color:#E9A13B;margin-top:12px">'
        "Demo mode — this result is synthetic, not a real prediction.</p>"
        if demo
        else ""
    )

    st.markdown(
        f'<div class="dr-panel" style="border-left:4px solid {colour};padding:26px 28px">'
        f'<p style="font-size:26px;font-weight:600;line-height:1.25;margin:0;'
        f'letter-spacing:-.02em;color:{colour}">{action}</p>'
        f'<p style="font-size:15.5px;color:#DDE7F1;margin-top:12px;max-width:62ch;'
        f'line-height:1.6">{detail}</p>{scale}{flag}</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# history
# ---------------------------------------------------------------------------
# Kept in session state only: it lives as long as the browser tab and is never
# written to disk. Retinal photographs are biometric identifiers, and the model
# card commits to storing them nowhere — an on-disk history would quietly break
# that promise.

HISTORY_LIMIT = 10


def _history() -> list[dict]:
    return st.session_state.setdefault("history", [])


def history_find(digest: str) -> dict | None:
    return next((entry for entry in _history() if entry["digest"] == digest), None)


def history_add(entry: dict) -> None:
    """Newest first, oldest dropped past the limit."""
    history = _history()
    history.insert(0, entry)
    del history[HISTORY_LIMIT:]


def history_clear() -> None:
    st.session_state["history"] = []
    st.session_state.pop("viewing", None)


def encode_jpeg(image_bgr: np.ndarray, quality: int = 80) -> bytes:
    """
    History entries hold JPEG bytes, not arrays.

    Ten entries of raw 384x384 arrays would be ~40 MB of session state; as JPEG
    it is under 2 MB, which matters on a 1 GB host where the ONNX session has
    already taken its share.
    """
    ok, buffer = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return buffer.tobytes() if ok else b""


def decode_jpeg(data: bytes) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def sidebar_history() -> str | None:
    """
    Render the reading history. Returns the digest the user asked to see again,
    or None.
    """
    history = _history()
    st.sidebar.markdown("**Recent readings**")

    if not history:
        st.sidebar.caption("Images you read appear here. Nothing is saved to disk.")
        return None

    selected = None
    viewing = st.session_state.get("viewing")

    for entry in history:
        current = entry["digest"] == viewing
        label = f"{STAGE_DOT[entry['grade']]} {entry['short_name']}"
        if st.sidebar.button(
            label,
            key=f"history-{entry['digest']}",
            width="stretch",
            type="primary" if current else "secondary",
            help=f"{entry['grade_name']} · {entry['action_label']} · {entry['time']}",
        ):
            selected = entry["digest"]
        st.sidebar.caption(f"　{entry['action_label']} · {entry['time']}")

    st.sidebar.caption(f"Kept in this tab only, newest {HISTORY_LIMIT}.")
    if st.sidebar.button("Clear history", key="history-clear", width="stretch"):
        history_clear()
        st.rerun()

    return selected


def likelihood_phrase(probability: float) -> tuple[str, str]:
    """A probability as a yes/no answer, with the colour that goes with it."""
    if probability >= 0.90:
        return "Yes, very likely", "#CF3D57"
    if probability >= 0.65:
        return "Yes, probably", "#E2663C"
    if probability >= 0.35:
        return "Unclear", "#E9A13B"
    if probability >= 0.10:
        return "Probably not", "#BFC94E"
    return "No", "#3FB98A"


def plain_answers(prediction) -> None:
    """
    The three screening questions, answered in words.

    Same numbers as the technical panel, read the way a screener would ask
    them. The percentages are one click away for anyone who wants them.
    """
    rows = []
    questions = [
        ("Any diabetic retinopathy?", prediction.probability_any_dr, False),
        ("Does this need a referral?", prediction.probability_referable, False),
        ("Is the photo good enough to read?", prediction.probability_ungradable, True),
    ]
    for question, probability, invert in questions:
        # the quality question reads better answered the positive way round
        answer, colour = likelihood_phrase(1 - probability if invert else probability)
        if invert:
            answer = {"Yes, very likely": "Yes, clearly", "Yes, probably": "Yes",
                      "Unclear": "Borderline", "Probably not": "Probably not",
                      "No": "No"}[answer]
            colour = {"#CF3D57": "#3FB98A", "#E2663C": "#3FB98A", "#E9A13B": "#E9A13B",
                      "#BFC94E": "#E2663C", "#3FB98A": "#CF3D57"}[colour]
        rows.append(
            f'<div style="display:flex;justify-content:space-between;gap:14px;'
            f'padding:11px 0;border-bottom:1px solid #1E2C3D">'
            f'<span style="font-size:13.5px;color:#97A9BD">{question}</span>'
            f'<span style="font-size:13.5px;font-weight:600;color:{colour};'
            f'white-space:nowrap">{answer}</span></div>'
        )
    st.markdown(
        f'<div class="dr-panel"><p style="font-weight:600;font-size:14.5px;'
        f'margin:0 0 6px">In short</p>{"".join(rows)}</div>',
        unsafe_allow_html=True,
    )
