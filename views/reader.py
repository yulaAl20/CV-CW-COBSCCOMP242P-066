"""
The reading page.

Laid out so the answer comes first. Someone screening a clinic list needs to
know what to do with this patient; the probability vector, the uncertainty
spread and the preprocessing stages are the evidence for that decision, so they
sit behind one reveal rather than competing with it.

This is a second front end over the same engine, not a second implementation.
Preprocessing, inference and the triage rules are imported from `backend/`,
which has no web framework in it — only `backend/main.py` knows about FastAPI.
"""

from __future__ import annotations

import hashlib
import time
from datetime import datetime

import cv2
import numpy as np
import streamlit as st

import streamlit_ui as ui
from backend import demo as demo_mode
from backend.config import settings
from backend.explain import class_activation_map, overlay_heatmap
from backend.preprocessing import (
    looks_like_fundus,
    preprocessing_stages,
    to_model_tensor,
)
from backend.triage import decide

ui.apply_theme()
ui.follow_navigation()
engine = ui.load_engine()

# ---------------------------------------------------------------------------
# sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("RetinaTriage")
st.sidebar.caption("Diabetic retinopathy stage detection")
ui.model_status(engine)

st.sidebar.divider()
ui.sidebar_history_button(ui.READER_PAGE)

st.sidebar.divider()
with st.sidebar.expander("Settings"):
    show_stages = st.checkbox("Keep each preprocessing step", value=True)
    show_heatmap = st.checkbox(
        "Work out the evidence heatmap",
        value=True,
        disabled=not engine.explains,
        help=None if engine.explains else "Needs models/backbone_cam.onnx.",
    )

with st.sidebar.expander("Decision thresholds"):
    st.markdown(
        f"""<div style="font-size:12.5px;color:#97A9BD;line-height:1.9">
        Re-capture above &nbsp;<span class="num">{settings.ungradable_threshold:.2f}</span><br>
        Defer above spread &nbsp;<span class="num">{settings.uncertainty_threshold:.2f}</span><br>
        Report above confidence &nbsp;<span class="num">{settings.confidence_threshold:.2f}</span><br>
        Refer at stage &nbsp;<span class="num">{settings.referral_grade}</span>
        </div>""",
        unsafe_allow_html=True,
    )
    st.caption("Set through environment variables. Read the model card first.")

# ---------------------------------------------------------------------------
# intake
# ---------------------------------------------------------------------------

if engine.is_ready and engine.is_stand_in:
    # These weights load cleanly and grade every image the same way, so the
    # app would otherwise look healthy while telling you nothing.
    st.error(
        "**These are stand-in weights, not a trained model.** "
        "`tests/make_fake_model.py` writes a random projection into `models/` so "
        "the tests can run without the 81 MB export. It returns a near-identical "
        "reading for every image. Delete the contents of `models/` and install "
        "the real export — see the README.",
        icon="🚫",
    )

if not engine.is_ready and not settings.allow_demo_mode:
    st.error(engine.load_error or "The model is not available.")
    st.stop()


def upload_box():
    return st.file_uploader(
        "Fundus photograph",
        type=["jpg", "jpeg", "png", "tif", "tiff"],
        label_visibility="collapsed",
        help=f"JPEG, PNG or TIFF · up to {settings.max_upload_mb} MB",
        # A new key gives an empty uploader. Opening a past scan from the history
        # page bumps it, otherwise the last upload still sitting in the box would
        # be taken as the image to show and replace the scan that was asked for.
        key=f"uploader-{st.session_state.get('uploader_round', 0)}",
    )


# Nothing read yet: the landing layout, with the upload box beside the retina.
# Once a scan is showing, the page drops to a compact header so the result
# comes first.
landing = ui.history_find(st.session_state.get("viewing", "")) is None

if landing:
    intro, picture = st.columns([1.05, 1], gap="large", vertical_alignment="center")
    with intro:
        st.markdown(
            '<h1 class="land-title">Read a fundus photograph</h1>'
            '<p class="land-lede">Upload a photo of the back of the eye to check it '
            "for diabetic retinopathy. You get the stage of disease, what to do with "
            "the patient next, and the evidence behind both.</p>",
            unsafe_allow_html=True,
        )
        upload = upload_box()
        st.markdown(
            '<p class="land-hint">Use the original camera capture. A screenshot or a '
            "crop loses the fine detail the model reads.</p>",
            unsafe_allow_html=True,
        )
    with picture:
        st.markdown(ui.retina_svg() + ui.severity_scale(settings.referral_grade),
                    unsafe_allow_html=True)
else:
    st.title("Read a fundus photograph")
    upload = upload_box()

# ---------------------------------------------------------------------------
# grading
# ---------------------------------------------------------------------------


def analyse(image: np.ndarray, digest: str, filename: str) -> dict:
    """Grade one photograph and package it as a history entry."""
    warnings: list[str] = []
    if not looks_like_fundus(image):
        warnings.append(
            "This does not look like a retinal fundus photograph, so the reading "
            "below is very unlikely to mean anything."
        )

    started = time.perf_counter()
    stages = preprocessing_stages(image, settings.image_size)
    preprocessed = stages["final"]
    preprocess_ms = (time.perf_counter() - started) * 1000

    tensor = to_model_tensor(preprocessed, settings.image_size)

    using_demo = not engine.is_ready or engine.is_stand_in
    if not engine.is_ready:
        prediction = demo_mode.synthetic_prediction(preprocessed, settings.mc_dropout_samples)
    else:
        prediction = engine.predict(tensor)

    decision = decide(prediction, settings)

    heatmap_bytes = None
    explain_ms = 0.0
    if show_heatmap and engine.explains:
        started = time.perf_counter()
        heatmap = class_activation_map(engine, tensor)
        if heatmap is not None:
            heatmap_bytes = ui.encode_jpeg(overlay_heatmap(preprocessed, heatmap))
        explain_ms = (time.perf_counter() - started) * 1000

    read_at = datetime.now()
    return {
        "digest": digest,
        "filename": filename,
        "short_name": (filename[:22] + "…") if len(filename) > 23 else filename,
        "read_at": read_at,
        "time": read_at.strftime("%H:%M"),
        "grade": prediction.grade,
        "grade_name": prediction.grade_name,
        "action_label": decision.label,
        "prediction": prediction,
        "decision": decision,
        "demo": using_demo,
        "warnings": warnings,
        "original": ui.encode_jpeg(cv2.resize(image, (settings.image_size,) * 2)),
        "preprocessed": ui.encode_jpeg(preprocessed),
        "heatmap": heatmap_bytes,
        "stages": (
            {name: ui.encode_jpeg(img, 70) for name, img in stages.items()}
            if show_stages
            else {}
        ),
        "timing": {
            "preprocess": preprocess_ms,
            "inference": prediction.inference_ms,
            "explain": explain_ms,
        },
    }


if upload is not None:
    raw = upload.getvalue()

    if len(raw) > settings.max_upload_mb * 1024 * 1024:
        st.error(f"That file is {len(raw) / 1e6:.1f} MB. The limit is {settings.max_upload_mb} MB.")
        st.stop()

    digest = hashlib.sha256(raw).hexdigest()[:16]
    # file_id changes on every upload event, so re-uploading the same picture
    # is still treated as a new action; the digest is the history identity.
    upload_key = getattr(upload, "file_id", None) or digest

    if upload_key != st.session_state.get("last_upload_key"):
        st.session_state["last_upload_key"] = upload_key
        st.session_state["viewing"] = digest

        if ui.history_find(digest) is None:
            image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                st.error(
                    f"'{upload.name}' could not be read as an image. Upload a JPEG, "
                    "PNG or TIFF fundus photograph."
                )
                st.stop()
            if min(image.shape[:2]) < 96:
                st.error(
                    "That image is smaller than 96 pixels on one side. Upload the "
                    "original capture."
                )
                st.stop()
            with st.spinner("Reading the photograph…"):
                ui.history_add(analyse(image, digest, upload.name))
            st.rerun()

entry = ui.history_find(st.session_state.get("viewing", ""))

if entry is None:
    st.markdown(
        ui.landing_steps(settings.mc_dropout_samples) + ui.landing_trust(ui.load_metrics()),
        unsafe_allow_html=True,
    )
    ui.disclaimer()
    st.stop()

prediction = entry["prediction"]
decision = entry["decision"]

# ---------------------------------------------------------------------------
# the answer
# ---------------------------------------------------------------------------

st.divider()

if len(ui._history()) > 1:
    st.caption(f"Showing **{entry['filename']}** · read at {entry['time']}")

ui.plain_summary(prediction, decision, demo=entry["demo"])

st.write("")
download, history, _ = st.columns([1, 1, 1.4])
with download:
    ui.download_report_button(entry, key=f"report-{entry['digest']}", primary=True)
with history:
    st.button("All scans", icon=":material/history:", width="stretch",
              key="reader-open-history", on_click=ui.go_to, args=(ui.HISTORY_PAGE,))

for message in entry["warnings"]:
    st.warning(message, icon="⚠️")

st.write("")
left, right = st.columns([1.15, 1], gap="large")

with left:
    labels = ["The photograph"]
    if entry["heatmap"]:
        labels.append("What the model looked at")
    tabs = st.tabs(labels)

    with tabs[0]:
        st.image(ui.rgb(ui.decode_jpeg(entry["original"])), width="stretch")

    if entry["heatmap"]:
        with tabs[1]:
            st.image(ui.rgb(ui.decode_jpeg(entry["heatmap"])), width="stretch")
            st.markdown(
                '<p class="dr-note">Warm areas are the ones that pushed the reading '
                "towards more severe disease. On a true positive they should sit on "
                "the damage, not on the bright disc in the middle or the edge of the "
                "picture.</p>",
                unsafe_allow_html=True,
            )

with right:
    ui.plain_answers(prediction)

# ---------------------------------------------------------------------------
# the working
# ---------------------------------------------------------------------------

st.write("")
with st.expander("Show the numbers behind this reading"):
    st.markdown("**Where it sits on the severity scale**")
    ui.severity_ladder(prediction)

    st.write("")
    scale, bars = st.columns(2, gap="large")

    with scale:
        ui.stage_bars(prediction)

    with bars:
        st.markdown(
            f"""<div class="dr-panel">
            <p style="font-weight:600;font-size:14.5px;margin:0 0 14px">Exact figures</p>
            <div style="font-size:13px;color:#97A9BD;line-height:2.1">
            Any diabetic retinopathy
              <span class="num" style="float:right;color:#DDE7F1">
              {prediction.probability_any_dr * 100:.1f}%</span><br>
            Referable disease, stage 2+
              <span class="num" style="float:right;color:#DDE7F1">
              {prediction.probability_referable * 100:.1f}%</span><br>
            Too poor to grade
              <span class="num" style="float:right;color:#DDE7F1">
              {prediction.probability_ungradable * 100:.1f}%</span><br>
            Continuous severity, 0–4
              <span class="num" style="float:right;color:#DDE7F1">
              {prediction.expected_grade:.2f}</span><br>
            Spread across {prediction.mc_samples} passes
              <span class="num" style="float:right;color:#DDE7F1">
              {prediction.uncertainty:.3f}</span><br>
            Entropy
              <span class="num" style="float:right;color:#DDE7F1">
              {prediction.entropy:.3f}</span>
            </div></div>""",
            unsafe_allow_html=True,
        )

    st.write("")
    st.markdown("**Why this decision**")
    st.markdown(
        f'<div class="dr-panel"><p style="font-size:13.5px;color:#97A9BD;margin:0">'
        f'{decision.reason}</p><p class="dr-note" style="margin-top:10px">'
        f"Rule {decision.rule} of 4 — quality first, then certainty, then "
        f"severity. The first rule that fires decides.</p></div>",
        unsafe_allow_html=True,
    )

    if entry["stages"]:
        st.write("")
        st.markdown("**What the photograph went through before grading**")
        captions = {
            "raw": "As uploaded",
            "cropped": "Cropped to the retina",
            "resized": "Resized to 384 px",
            "clahe": "Contrast enhanced",
            "illumination": "Illumination corrected",
            "final": "Masked — what the model sees",
        }
        ordered = [key for key in captions if key in entry["stages"]]
        for start in range(0, len(ordered), 3):
            for column, key in zip(st.columns(3), ordered[start:start + 3], strict=False):
                with column:
                    st.image(
                        ui.rgb(ui.decode_jpeg(entry["stages"][key])),
                        caption=captions[key],
                        width="stretch",
                    )

    timing = entry["timing"]
    total = sum(timing.values())
    st.caption(
        f"{total:.0f} ms on CPU — {timing['preprocess']:.0f} ms preparing the image, "
        f"{timing['inference']:.0f} ms reading it"
        + (f", {timing['explain']:.0f} ms working out the heatmap" if timing["explain"] else "")
    )

st.divider()
st.info(
    "**Before you trust any of this.** Every reading comes from a model that gets "
    "some cases wrong. **Model evidence** in the sidebar shows exactly which ones, "
    "measured on images it never saw during training.",
    icon="📊",
)

ui.disclaimer()
