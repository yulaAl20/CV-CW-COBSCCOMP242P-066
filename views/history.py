

from __future__ import annotations

import csv
import io
from datetime import datetime

import streamlit as st

import streamlit_ui as ui

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
ui.sidebar_history_button(ui.HISTORY_PAGE)

# ---------------------------------------------------------------------------
# page
# ---------------------------------------------------------------------------

history = ui._history()

st.title("Scan history")
st.markdown(
    '<p class="dr-lede">Every photograph read in this browser tab, newest first. '
    "Open any of them again, or download its report as a PDF to keep with the "
    "patient's record.</p>",
    unsafe_allow_html=True,
)

if not history:
    st.info("No scans yet. Read a fundus photograph and it will appear here.",
            icon="🗂️")
    st.button("Read an image", icon=":material/visibility:", type="primary",
              on_click=ui.go_to, args=(ui.READER_PAGE,))
    ui.disclaimer()
    st.stop()


def _when(entry: dict) -> str:
    read_at = entry.get("read_at")
    return read_at.strftime("%d %b %Y, %H:%M") if read_at else entry.get("time", "")


# --- the session at a glance ------------------------------------------------

counts = dict.fromkeys(("ophthalmology", "human", "recapture", "routine"), 0)
for entry in history:
    counts[entry["decision"].action] = counts.get(entry["decision"].action, 0) + 1

for column, (value, label, sub) in zip(
    st.columns(4),
    [
        (counts["ophthalmology"], "Refer to a specialist", "referable disease"),
        (counts["human"], "Needs a human grader", "model not sure enough"),
        (counts["recapture"], "Re-capture", "photo too poor to read"),
        (counts["routine"], "Routine rescreen", "no referral needed"),
    ],
    strict=True,
):
    with column:
        ui.tile(str(value), label, sub)

st.write("")

# --- whole-session actions --------------------------------------------------


def _session_csv() -> bytes:
    """One row per scan, for a spreadsheet. Built in memory, never on disk."""
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow([
        "file", "read_at", "stage", "stage_name", "next_step", "confidence_%",
        "any_dr_%", "referable_%", "ungradable_%", "uncertainty", "rule",
        "report_id", "synthetic_demo_result",
    ])
    for entry in history:
        prediction, decision = entry["prediction"], entry["decision"]
        read_at = entry.get("read_at")
        writer.writerow([
            entry["filename"],
            read_at.isoformat(timespec="seconds") if read_at else entry.get("time", ""),
            prediction.grade,
            prediction.grade_name,
            decision.label,
            f"{prediction.confidence * 100:.1f}",
            f"{prediction.probability_any_dr * 100:.1f}",
            f"{prediction.probability_referable * 100:.1f}",
            f"{prediction.probability_ungradable * 100:.1f}",
            f"{prediction.uncertainty:.3f}",
            decision.rule,
            entry["digest"],
            "yes" if entry.get("demo") else "no",
        ])
    return buffer.getvalue().encode("utf-8")


left, middle, _ = st.columns([1.2, 1, 1.6])
with left:
    st.download_button(
        "Download all (CSV)",
        data=_session_csv(),
        file_name=f"retinatriage_session_{datetime.now():%Y%m%d_%H%M}.csv",
        mime="text/csv",
        icon=":material/table_view:",
        on_click="ignore",
        width="stretch",
        key="history-csv",
    )
with middle:
    st.button("Clear history", icon=":material/delete:", width="stretch",
              key="history-clear-page", on_click=ui.history_clear)

st.write("")

# --- one card per scan ------------------------------------------------------

for entry in history:
    prediction, decision = entry["prediction"], entry["decision"]
    colour = ui.URGENCY_COLOURS.get(decision.urgency, "#8298AE")

    with st.container(border=True):
        picture, details, actions = st.columns([0.9, 3, 1.3], vertical_alignment="center")

        with picture:
            st.image(ui.rgb(ui.decode_jpeg(entry["original"])), width="stretch")

        with details:
            stage = (
                "Not graded — photo too poor"
                if decision.action == "recapture"
                else f"{ui.STAGE_DOT[prediction.grade]} Stage {prediction.grade} · "
                     f"{prediction.grade_name}"
            )
            demo = (
                ' · <span style="color:#E9A13B">synthetic demo result</span>'
                if entry.get("demo") else ""
            )
            st.markdown(
                f'<p style="font-weight:600;font-size:15.5px;margin:0;'
                f'overflow-wrap:anywhere">{entry["filename"]}</p>'
                f'<p style="font-size:12.5px;color:#6B7F95;margin:2px 0 10px">'
                f"{_when(entry)}{demo}</p>"
                f'<p style="font-size:14px;margin:0">{stage}</p>'
                f'<p style="font-size:14px;font-weight:600;color:{colour};margin:4px 0 0">'
                f"{decision.label}</p>"
                f'<p style="font-size:12.5px;color:#97A9BD;margin:2px 0 0">'
                f"{prediction.confidence * 100:.0f}% confidence in the stage</p>",
                unsafe_allow_html=True,
            )

        with actions:
            st.button("View", icon=":material/visibility:", width="stretch",
                      key=f"view-{entry['digest']}", on_click=ui.view_scan,
                      args=(entry["digest"],))
            ui.download_report_button(entry, key=f"pdf-{entry['digest']}", label="PDF")

st.caption(
    f"Kept in this browser tab only, newest {ui.HISTORY_LIMIT}. Closing or "
    "refreshing the tab clears it — download the reports you want to keep."
)
ui.disclaimer()
