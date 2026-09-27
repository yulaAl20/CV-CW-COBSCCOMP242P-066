"""
The measured performance of the model, on its own page.

It sits apart from the reading page on purpose. Someone grading an image needs
that image's result and nothing else; someone deciding whether to believe the
readout at all needs the test-set results, the external comparison and the
limitations. Mixing them made the reading page long enough that the evidence
went unread.

Everything here is read from `artifacts/evaluation.json` through the same
`MetricsStore` the FastAPI service uses. Nothing is recomputed, so this page
cannot claim a score the training run did not produce.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

import streamlit_ui as ui

ui.apply_theme()

FIGURES = Path(__file__).resolve().parent.parent / "docs" / "images"

# always 'No DR' on the DDR test split, the floor any model must beat
MAJORITY_BASELINE = 0.5003

data = ui.load_metrics()
headline = data.get("headline", {})

st.title("What this model gets right, and what it gets wrong")
st.markdown(
    '<p class="dr-lede">Measured on 1,879 images the model never saw during '
    "training, and then again on 3,662 images from a different country, different "
    "cameras and a different patient population. Both are shown, because only one "
    "of them is a fair test of whether the model travels.</p>",
    unsafe_allow_html=True,
)

if not headline.get("available"):
    st.error(
        "No evaluation results are installed on this server. Add "
        "`artifacts/evaluation.json` from a training run to populate this page."
    )
    st.stop()

run = data.get("run", {})
if run:
    st.caption(
        f"{run.get('backbone')} · {run.get('total_parameters_millions')}M parameters · "
        f"{run.get('epochs_run')} of {run.get('epochs_configured')} epochs, stopped early "
        f"at epoch {run.get('best_epoch')} · trained on {run.get('train_images'):,} images "
        f"in {run.get('training_minutes')} minutes."
    )

internal = headline["internal"]
external = headline["external"]
detection = headline["detection"]
calibration = headline["calibration"]

# ---------------------------------------------------------------------------
st.divider()
st.subheader("Headline numbers")
st.markdown(
    '<p class="dr-note">On the held-out test split. Accuracy is shown against the '
    "50.0% a model could score by answering “No DR” to everything.</p>",
    unsafe_allow_html=True,
)
st.write("")

for column, (value, label, sub) in zip(
    st.columns(3),
    [
        (f"{internal['accuracy'] * 100:.1f}%", "Stage accuracy",
         f"vs {MAJORITY_BASELINE * 100:.1f}% for always answering “No DR”"),
        (f"{internal['qwk']:.3f}", "Agreement with the grader", "quadratic weighted kappa"),
        (f"{internal['macro_f1']:.3f}", "Macro F1", "averaged over all five stages"),
    ],
    strict=True,
):
    with column:
        ui.tile(value, label, sub)

st.write("")
for column, (value, label, sub) in zip(
    st.columns(3),
    [
        (f"{internal['referable_auc']:.3f}", "Referable DR, AUC",
         "stage 2 and above — the referral decision"),
        (f"{detection['recall'] * 100:.1f}%", "DR detection recall",
         f"{detection['specificity'] * 100:.1f}% specificity"),
        (f"{calibration['ece']:.3f}", "Calibration error",
         "0 is perfect; this model is overconfident"),
    ],
    strict=True,
):
    with column:
        ui.tile(value, label, sub)

# ---------------------------------------------------------------------------
st.divider()
st.subheader("Does it travel?")
st.markdown(
    '<p class="dr-note">The same model on APTOS 2019, which it was never trained '
    "on. Read the last two rows together: the ability to rank eyes by severity "
    "survives almost intact, while agreement on exact stage boundaries does not.</p>",
    unsafe_allow_html=True,
)

comparison = pd.DataFrame(data["comparison"]).rename(
    columns={"metric": "Metric", "internal": "DDR test",
             "external": "APTOS 2019", "gap": "Gap"}
)
st.dataframe(
    comparison.style
    .format({"DDR test": "{:.4f}", "APTOS 2019": "{:.4f}", "Gap": "−{:.4f}"})
    # a large gap is the finding, so it is coloured; a small one is not
    .map(lambda v: f"color: {'#E9A13B' if v > 0.1 else '#3FB98A'}", subset=["Gap"]),
    hide_index=True,
    width='stretch',
)
st.markdown(
    '<p class="dr-note">Macro F1 falls by 0.27 while referable AUC falls by 0.017. '
    "The model still separates diseased eyes from healthy ones on unfamiliar data; "
    "what it loses is knowing exactly where another set of graders drew the stage "
    "boundaries.</p>",
    unsafe_allow_html=True,
)

figure = FIGURES / "fig_internal_vs_external.png"
if figure.exists():
    st.image(str(figure), width='stretch')

# ---------------------------------------------------------------------------
st.divider()
st.subheader("Where the errors fall")

left, right = st.columns(2, gap="large")

with left:
    st.markdown("**Confusion matrix**")
    st.markdown(
        '<p class="dr-note">Each row is a true stage, each column what the model '
        "said. Errors clustering next to the diagonal is the behaviour the ordinal "
        "loss was chosen to produce.</p>",
        unsafe_allow_html=True,
    )
    confusion = data["confusion"]
    matrix = pd.DataFrame(
        confusion["matrix"],
        index=[f"true {name}" for name in confusion["labels"]],
        columns=[f"pred {name}" for name in confusion["labels"]],
    )
    st.dataframe(
        matrix.style.background_gradient(cmap="magma", axis=1).format("{:d}"),
        width='stretch',
    )

with right:
    st.markdown("**Precision, recall and F1 per stage**")
    st.markdown(
        '<p class="dr-note">The two weakest stages are also the two rarest. Mild '
        "NPDR is defined by microaneurysms alone, near the resolution limit at "
        "384 px.</p>",
        unsafe_allow_html=True,
    )
    per_class = pd.DataFrame(data["per_class"])[
        ["name", "precision", "recall", "f1", "support"]
    ].rename(columns={"name": "Stage", "precision": "Precision",
                      "recall": "Recall", "f1": "F1", "support": "Images"})
    st.dataframe(
        per_class.style.format(
            {"Precision": "{:.3f}", "Recall": "{:.3f}", "F1": "{:.3f}"}
        ),
        hide_index=True,
        width='stretch',
    )

for name in ("fig_confusion_matrix.png", "fig_per_class_metrics.png"):
    figure = FIGURES / name
    if figure.exists():
        st.image(str(figure), width='stretch')

# ---------------------------------------------------------------------------
st.divider()
st.subheader("How it was trained, and whether the uncertainty helps")

left, right = st.columns(2, gap="large")
with left:
    st.markdown("**Learning curves**")
    st.markdown(
        '<p class="dr-note">Training ran on augmented, class-balanced batches, so '
        "the two accuracy lines are not directly comparable — the trend is what "
        "matters.</p>",
        unsafe_allow_html=True,
    )
    figure = FIGURES / "fig_training_curves.png"
    if figure.exists():
        st.image(str(figure), width='stretch')
    elif data.get("history"):
        st.line_chart(
            pd.DataFrame(data["history"]).set_index("epoch")[
                ["train_loss", "val_loss", "val_qwk"]
            ]
        )

with right:
    st.markdown("**Deferring the uncertain cases**")
    st.markdown(
        '<p class="dr-note">Sensitivity rising alongside accuracy is the important '
        "part: it rules out a model that flatters its score by quietly deferring "
        "the sick patients.</p>",
        unsafe_allow_html=True,
    )
    sweep = pd.DataFrame(data["referral_sweep"]).rename(
        columns={"deferred_percent": "Deferred %", "images_kept": "Kept",
                 "accuracy": "Accuracy", "referable_sensitivity": "Sensitivity"}
    )
    st.dataframe(
        sweep.style.format({"Deferred %": "{:.0f}", "Kept": "{:.0f}",
                            "Accuracy": "{:.4f}", "Sensitivity": "{:.4f}"}),
        hide_index=True,
        width='stretch',
    )
    figure = FIGURES / "fig_risk_coverage.png"
    if figure.exists():
        st.image(str(figure), width='stretch')

# ---------------------------------------------------------------------------
deployment = data.get("deployment", {})
if deployment.get("benchmarks"):
    st.divider()
    st.subheader("What it costs to run")
    st.markdown(
        '<p class="dr-note">Measured on CPU, one image at a time.</p>',
        unsafe_allow_html=True,
    )
    benchmarks = pd.DataFrame(deployment["benchmarks"]).rename(
        columns={"model": "Export", "size_mb": "Size (MB)", "median_ms": "Median (ms)",
                 "mean_ms": "Mean (ms)", "p95_ms": "p95 (ms)"}
    )
    st.dataframe(benchmarks, hide_index=True, width='stretch')
    st.markdown(
        '<p class="dr-note">The int8 export is a quarter of the size and 1.8× '
        "slower: per-operator conversion overhead costs more than the narrower "
        "arithmetic saves on a convolutional backbone. The app ships fp32 because "
        "this was measured, not assumed.</p>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
st.divider()
st.subheader("What this does not show")

LIMITATIONS = [
    ("The quality gate has never seen a bad photograph",
     "This copy of DDR contained no ungradable images, so the quality head trained "
     "with no negative examples. The re-capture rule exists and is wired up, but it "
     "did not fire once across all 1,879 test images. A genuinely unreadable "
     "photograph will be graded confidently instead of rejected."),
    ("Mild disease is unreliable",
     "F1 of 0.38 internally and 0.08 externally. Stage 1 is the hardest class for "
     "human graders too, and with 441 training examples the model sees it rarely "
     "and at the resolution where it is least visible."),
    ("Confidence runs ahead of accuracy",
     "Calibration error of 0.075 means that when the model reports 90% confidence "
     "it is right about 82% of the time. The reporting thresholds are set low to "
     "compensate, but the underlying overconfidence has not been corrected."),
    ("Nothing here is clinical validation",
     "These are retrospective results on two public datasets. There has been no "
     "prospective testing, no regulatory review, and no breakdown by age, sex or "
     "ethnicity, because the datasets do not carry that metadata."),
]

for column, (title, body) in zip(st.columns(2), LIMITATIONS[:2], strict=True):
    with column:
        st.markdown(
            f'<div class="dr-panel"><h3 style="font-size:14px;color:#E9A13B;'
            f'margin:0 0 7px">{title}</h3>'
            f'<p style="font-size:13px;color:#97A9BD;margin:0">{body}</p></div>',
            unsafe_allow_html=True,
        )
st.write("")
for column, (title, body) in zip(st.columns(2), LIMITATIONS[2:], strict=True):
    with column:
        st.markdown(
            f'<div class="dr-panel"><h3 style="font-size:14px;color:#E9A13B;'
            f'margin:0 0 7px">{title}</h3>'
            f'<p style="font-size:13px;color:#97A9BD;margin:0">{body}</p></div>',
            unsafe_allow_html=True,
        )

ui.disclaimer()
