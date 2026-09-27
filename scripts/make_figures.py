"""
Generate every figure used in the report and the README.

    python scripts/make_figures.py

    fig_architecture_system     how a request flows through the deployed app
    fig_architecture_model      the network and its two heads
    fig_pipeline_preprocessing  the six preprocessing steps
    fig_triage_flow             the four triage rules as a decision tree
    fig_confusion_matrix        stage-level errors on the DDR test set
    fig_per_class_metrics       precision / recall / F1 per stage
    fig_training_curves         loss, accuracy and agreement by epoch
    fig_internal_vs_external    the generalisation gap
    fig_risk_coverage           accuracy as uncertain cases are deferred
    fig_triage_distribution     what the policy does to each true stage
    fig_class_imbalance         why accuracy alone is not enough
    fig_cpu_latency             export size against CPU speed
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = PROJECT_ROOT / "artifacts"
OUTPUT = PROJECT_ROOT / "docs" / "images"

# The same palette the interface uses, so the report and the running app look
# like one project rather than two.
INK = "#0E1620"
PANEL = "#16212E"
PANEL_HIGH = "#1C2A3A"
LINE = "#2B3D53"
TEXT = "#DDE7F1"
DIM = "#97A9BD"
FAINT = "#6B7F95"
ACCENT = "#5B8DEF"

GRADE_COLOURS = ["#3FB98A", "#BFC94E", "#E9A13B", "#E2663C", "#CF3D57"]
GRADE_SHORT = ["No DR", "Mild", "Moderate", "Severe", "PDR"]
GRADE_FULL = ["No DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "Proliferative DR"]

plt.rcParams.update({
    "figure.facecolor": INK,
    "axes.facecolor": PANEL,
    "savefig.facecolor": INK,
    "text.color": TEXT,
    "axes.labelcolor": DIM,
    "axes.edgecolor": LINE,
    "xtick.color": FAINT,
    "ytick.color": FAINT,
    "grid.color": "#223145",
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.titleweight": "semibold",
    "axes.grid": True,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,
    "figure.dpi": 160,
})


def save(fig, name: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / f"{name}.png"
    fig.savefig(path, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"  {path.relative_to(PROJECT_ROOT)}")


def load_evaluation() -> dict:
    with (ARTIFACTS / "evaluation.json").open() as file:
        return json.load(file)


# ===========================================================================
# diagram primitives
# ===========================================================================


def box(ax, x, y, width, height, label, sublabel="", colour=PANEL_HIGH,
        edge=LINE, text_colour=TEXT, fontsize=8.5):
    ax.add_patch(FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.1, facecolor=colour, edgecolor=edge, zorder=2,
    ))
    if sublabel:
        ax.text(x + width / 2, y + height * 0.62, label, ha="center", va="center",
                fontsize=fontsize, color=text_colour, weight="semibold", zorder=3)
        ax.text(x + width / 2, y + height * 0.28, sublabel, ha="center", va="center",
                fontsize=fontsize - 1.7, color=FAINT, zorder=3)
    else:
        ax.text(x + width / 2, y + height / 2, label, ha="center", va="center",
                fontsize=fontsize, color=text_colour, weight="semibold", zorder=3)


def arrow(ax, start, end, colour=ACCENT, label="", style="-|>", curve=0.0,
          label_offset=(0.0, 0.022)):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle=style, mutation_scale=16,
        linewidth=1.3, color=colour, zorder=4,
        shrinkA=0, shrinkB=0,
        connectionstyle=f"arc3,rad={curve}",
    ))
    if label:
        ax.text((start[0] + end[0]) / 2 + label_offset[0],
                (start[1] + end[1]) / 2 + label_offset[1], label,
                ha="center", va="bottom", fontsize=6.8, color=FAINT, zorder=5)


def blank_axes(width, height):
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor(INK)
    fig.patch.set_facecolor(INK)
    return fig, ax


# ===========================================================================
# architecture diagrams
# ===========================================================================


def figure_system_architecture() -> None:
    fig, ax = blank_axes(11, 5.4)
    ax.text(0.5, 0.965, "Deployed system: one container, one process",
            ha="center", fontsize=12.5, color=TEXT, weight="semibold")
    ax.text(0.5, 0.925, "FastAPI serves the interface and the JSON API from the same "
                        "origin, so there is no CORS layer and no second service to secure.",
            ha="center", fontsize=7.8, color=FAINT)

    # client
    box(ax, 0.02, 0.60, 0.17, 0.17, "Browser", "single-page interface", "#1A2637")
    box(ax, 0.02, 0.36, 0.17, 0.17, "Fundus camera", "JPEG / PNG / TIFF", "#1A2637")

    # container boundary
    ax.add_patch(FancyBboxPatch(
        (0.245, 0.14), 0.52, 0.70,
        boxstyle="round,pad=0.014,rounding_size=0.02",
        linewidth=1.1, facecolor="#131E2B", edgecolor="#33465E",
        linestyle="--", zorder=0,
    ))
    ax.text(0.505, 0.875, "Docker container  ·  python:3.11-slim  ·  CPU only",
            ha="center", fontsize=7.6, color=FAINT)

    box(ax, 0.265, 0.62, 0.20, 0.15, "FastAPI", "routing, validation", "#1E2B3C")
    box(ax, 0.265, 0.42, 0.20, 0.15, "Preprocessing", "OpenCV, 6 steps", "#1E2B3C")
    box(ax, 0.265, 0.22, 0.20, 0.15, "Triage rules", "4-step policy", "#1E2B3C")

    box(ax, 0.545, 0.62, 0.20, 0.15, "ONNX Runtime", "backbone, 81 MB", "#1E2B3C")
    box(ax, 0.545, 0.42, 0.20, 0.15, "MC dropout heads", "NumPy, 20 passes", "#1E2B3C")
    box(ax, 0.545, 0.22, 0.20, 0.15, "CAM explainer", "gradient-free", "#1E2B3C")

    # artifacts
    box(ax, 0.815, 0.52, 0.165, 0.17, "models/", "ONNX + heads.npz", "#1A2637")
    box(ax, 0.815, 0.28, 0.165, 0.17, "artifacts/", "recorded metrics", "#1A2637")

    arrow(ax, (0.19, 0.685), (0.265, 0.695))
    arrow(ax, (0.19, 0.445), (0.265, 0.495))
    arrow(ax, (0.465, 0.695), (0.545, 0.695))
    arrow(ax, (0.365, 0.62), (0.365, 0.575), colour=LINE)
    arrow(ax, (0.465, 0.495), (0.545, 0.495))
    arrow(ax, (0.645, 0.42), (0.645, 0.375), colour=LINE)
    arrow(ax, (0.545, 0.295), (0.47, 0.295), colour=LINE)
    # both artifact folders are read once at start-up, so they share one arrow
    # into the container rather than each crossing the diagram
    ax.plot([0.800, 0.800], [0.365, 0.605], color=LINE, linewidth=1.1, zorder=1)
    ax.plot([0.800, 0.815], [0.605, 0.605], color=LINE, linewidth=1.1, zorder=1)
    ax.plot([0.800, 0.815], [0.365, 0.365], color=LINE, linewidth=1.1, zorder=1)
    arrow(ax, (0.800, 0.485), (0.768, 0.485), colour=LINE)
    ax.text(0.783, 0.50, "read once\nat start-up", ha="center", va="bottom",
            fontsize=6.4, color=FAINT)

    # the return path curves below the input boxes rather than across them
    arrow(ax, (0.265, 0.275), (0.105, 0.60), colour="#3FB98A", curve=0.34)
    ax.text(0.128, 0.285, "JSON result", ha="center", fontsize=6.8, color="#3FB98A")

    ax.text(0.505, 0.055,
            "Request path:  upload  →  validate  →  preprocess  →  backbone (once)  →  "
            "heads × 20  →  triage  →  heatmap  →  JSON",
            ha="center", fontsize=7.6, color=DIM)
    save(fig, "fig_architecture_system")


def figure_model_architecture() -> None:
    fig, ax = blank_axes(11, 4.4)
    ax.text(0.5, 0.955, "Network: one shared backbone, two heads",
            ha="center", fontsize=12.5, color=TEXT, weight="semibold")
    ax.text(0.5, 0.905, "EfficientNetV2-S pretrained on ImageNet-21k · 20.2M parameters · "
                        "384 × 384 input · the heads add 7,686",
            ha="center", fontsize=7.8, color=FAINT)

    box(ax, 0.015, 0.45, 0.13, 0.20, "Fundus image", "384 × 384 × 3", "#1A2637")
    box(ax, 0.175, 0.45, 0.22, 0.20, "EfficientNetV2-S", "frozen epoch 0, then fine-tuned",
        "#243349", edge=ACCENT)
    box(ax, 0.425, 0.45, 0.13, 0.20, "Features", "1280-dim", "#1A2637")
    box(ax, 0.585, 0.45, 0.10, 0.20, "Dropout", "p = 0.4", "#1A2637")

    box(ax, 0.72, 0.62, 0.155, 0.17, "Grade head", "4 ordinal logits", "#243349")
    box(ax, 0.72, 0.32, 0.155, 0.17, "Quality head", "2 logits", "#243349")

    box(ax, 0.90, 0.62, 0.09, 0.17, "Stage", "0 – 4", GRADE_COLOURS[2], text_colour=INK)
    box(ax, 0.90, 0.32, 0.09, 0.17, "Gradable?", "yes / no", "#1A2637")

    arrow(ax, (0.145, 0.55), (0.175, 0.55))
    arrow(ax, (0.395, 0.55), (0.425, 0.55))
    arrow(ax, (0.555, 0.55), (0.585, 0.55))
    arrow(ax, (0.685, 0.58), (0.72, 0.70))
    arrow(ax, (0.685, 0.52), (0.72, 0.41))
    arrow(ax, (0.875, 0.705), (0.90, 0.705))
    arrow(ax, (0.875, 0.405), (0.90, 0.405))

    ax.text(0.285, 0.36, "transfer learning\nbackbone lr 1e-4, heads lr 1e-3",
            ha="center", fontsize=7, color=FAINT)
    ax.text(0.635, 0.23, "dropout stays on at inference:\n20 passes give the uncertainty",
            ha="center", fontsize=7, color="#E9A13B")

    ax.text(0.5, 0.11,
            "Ordinal (CORN) head:  four yes/no questions — \"worse than 0?\", \"worse than 1?\", "
            "\"worse than 2?\", \"worse than 3?\"",
            ha="center", fontsize=8, color=TEXT)
    ax.text(0.5, 0.055,
            "Chaining them with a cumulative product makes the answers impossible to "
            "contradict, and makes a two-stage error cost more than a one-stage error.",
            ha="center", fontsize=7.4, color=FAINT)
    save(fig, "fig_architecture_model")


def figure_preprocessing_pipeline() -> None:
    fig, ax = blank_axes(12, 3.0)
    ax.text(0.5, 0.93, "Preprocessing: six steps, applied identically in training and serving",
            ha="center", fontsize=12, color=TEXT, weight="semibold")

    steps = [
        ("Crop", "to the retina", "removes ~25% dead pixels"),
        ("Pad", "to a square", "no lesion distortion"),
        ("Resize", "384 × 384", "one size for all cameras"),
        ("CLAHE", "green channel", "contrast where lesions show"),
        ("Subtract", "local average", "flattens flash gradient 88%"),
        ("Mask", "circular", "drops the bright FOV rim"),
    ]

    width = 0.142
    gap = 0.025
    start = (1 - (len(steps) * width + (len(steps) - 1) * gap)) / 2

    for index, (title, subtitle, note) in enumerate(steps):
        x = start + index * (width + gap)
        box(ax, x, 0.42, width, 0.26, title, subtitle, "#1E2B3C", fontsize=9)
        ax.text(x + width / 2, 0.33, note, ha="center", va="top",
                fontsize=6.5, color=FAINT, wrap=True)
        if index < len(steps) - 1:
            arrow(ax, (x + width, 0.55), (x + width + gap, 0.55))

    ax.text(0.5, 0.13,
            "Serving reuses the exact functions from the training notebook. If these two "
            "ever diverge, every metric in this report stops describing the deployed app.",
            ha="center", fontsize=7.6, color="#E9A13B")
    save(fig, "fig_pipeline_preprocessing")


def figure_triage_flow() -> None:
    fig, ax = blank_axes(9.5, 5.6)
    ax.text(0.5, 0.965, "Triage policy: quality, then certainty, then severity",
            ha="center", fontsize=12.5, color=TEXT, weight="semibold")
    ax.text(0.5, 0.925, "Rules are checked in order. The first one that fires decides, "
                        "and the app reports which one it was.",
            ha="center", fontsize=7.8, color=FAINT)

    decisions = [
        (0.755, "1", "P(ungradable) > 0.50?", "recapture", "Re-capture image", "#8298AE"),
        (0.555, "2", "spread > 0.45  or  confidence < 0.55?", "human",
         "Refer for human grading", "#E9A13B"),
        (0.355, "3", "predicted stage ≥ 2?", "ophthalmology",
         "Refer to ophthalmology", "#CF3D57"),
    ]

    for y, number, question, _key, action, colour in decisions:
        box(ax, 0.10, y, 0.44, 0.115, f"Rule {number}   {question}", "", "#1E2B3C")
        box(ax, 0.63, y, 0.31, 0.115, action, "", colour, text_colour=INK)
        arrow(ax, (0.54, y + 0.058), (0.63, y + 0.058), colour=colour, label="yes")
        ax.text(0.32, y - 0.035, "no", ha="center", fontsize=6.8, color=FAINT)

    box(ax, 0.63, 0.155, 0.31, 0.115, "Routine rescreen", "", GRADE_COLOURS[0],
        text_colour=INK)
    box(ax, 0.10, 0.155, 0.44, 0.115, "Rule 4   everything else", "", "#1E2B3C")
    arrow(ax, (0.54, 0.213), (0.63, 0.213), colour=GRADE_COLOURS[0])

    arrow(ax, (0.32, 0.755), (0.32, 0.675), colour=LINE)
    arrow(ax, (0.32, 0.555), (0.32, 0.475), colour=LINE)
    arrow(ax, (0.32, 0.355), (0.32, 0.275), colour=LINE)

    ax.text(0.5, 0.065,
            "Ordering matters clinically: an unreadable photograph of a proliferative eye "
            "is still an unreadable photograph, so quality is checked before severity.",
            ha="center", fontsize=7.6, color=DIM)
    save(fig, "fig_triage_flow")


# ===========================================================================
# evaluation figures
# ===========================================================================


def figure_confusion_matrix(evaluation: dict) -> None:
    counts = np.array(evaluation["ddr_test"]["confusion_matrix"], dtype=float)
    shares = counts / counts.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    ax.grid(False)
    image = ax.imshow(shares, cmap="magma", vmin=0, vmax=1)

    ax.set_xticks(range(5), GRADE_SHORT, rotation=35, ha="right")
    ax.set_yticks(range(5), GRADE_SHORT)
    ax.set_xlabel("predicted stage")
    ax.set_ylabel("true stage")
    ax.set_title("Where the model's stage errors fall\nDDR held-out test, 1,879 images")

    for row in range(5):
        for column in range(5):
            ax.text(column, row, f"{shares[row, column]:.2f}\n({int(counts[row, column])})",
                    ha="center", va="center", fontsize=8,
                    color="white" if shares[row, column] < 0.5 else INK,
                    weight="semibold" if row == column else "normal")

    bar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    bar.set_label("share of the true stage", color=DIM, fontsize=8)
    bar.ax.tick_params(colors=FAINT)
    bar.outline.set_edgecolor(LINE)

    breakdown = evaluation["ddr_test"]["error_breakdown"]
    fig.text(0.5, -0.02,
             f"{breakdown['total_errors']} errors total: "
             f"{breakdown['off_by_one_stage']} off by one stage, "
             f"{breakdown['off_by_two_or_more']} off by two or more. "
             "Neighbouring-stage confusion is expected — human graders disagree there too.",
             ha="center", fontsize=7.6, color=FAINT)
    save(fig, "fig_confusion_matrix")


def figure_per_class_metrics(evaluation: dict) -> None:
    internal = evaluation["ddr_test"]
    precision = internal["precision_per_class"]
    recall = internal["recall_per_class"]
    f1 = internal["f1_per_class"]
    support = internal["support_per_class"]

    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, figsize=(8.6, 6.2), gridspec_kw={"height_ratios": [2.4, 1]}
    )

    positions = np.arange(5)
    width = 0.26
    for offset, (values, label, colour) in enumerate([
        (precision, "precision", ACCENT),
        (recall, "recall", "#3FB98A"),
        (f1, "F1", "#E9A13B"),
    ]):
        bars = ax_top.bar(positions + (offset - 1) * width, values, width,
                          label=label, color=colour, edgecolor="none")
        for bar, value in zip(bars, values, strict=True):
            ax_top.text(bar.get_x() + bar.get_width() / 2, value + 0.018,
                        f"{value:.2f}", ha="center", fontsize=7, color=DIM)

    ax_top.set_xticks(positions, GRADE_FULL, fontsize=8)
    ax_top.set_ylim(0, 1.12)
    ax_top.set_ylabel("score")
    ax_top.legend(frameon=False, ncol=3, loc="upper center",
                  labelcolor=DIM, fontsize=8)
    ax_top.set_title("Per-stage performance on the DDR test set")

    bars = ax_bottom.bar(positions, support, color=GRADE_COLOURS, edgecolor="none")
    for bar, value in zip(bars, support, strict=True):
        ax_bottom.text(bar.get_x() + bar.get_width() / 2, value + 14, str(value),
                       ha="center", fontsize=7.5, color=DIM)
    ax_bottom.set_xticks(positions, GRADE_SHORT, fontsize=8)
    ax_bottom.set_ylabel("test images")
    ax_bottom.set_ylim(0, max(support) * 1.2)

    fig.text(0.5, -0.015,
             "The two weakest stages, Mild and Severe, are also the two rarest. "
             "With 35 severe cases in the test set a single image moves recall by 3 points.",
             ha="center", fontsize=7.8, color="#E9A13B")
    fig.tight_layout()
    save(fig, "fig_per_class_metrics")


def figure_training_curves() -> None:
    history = pd.read_csv(ARTIFACTS / "history.csv")

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))

    axes[0].plot(history.epoch, history.train_loss, "o-", ms=3, color=ACCENT, label="train")
    axes[0].plot(history.epoch, history.val_loss, "o-", ms=3, color="#E2663C", label="validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend(frameon=False, labelcolor=DIM, fontsize=8)

    axes[1].plot(history.epoch, history.train_accuracy, "o-", ms=3, color=ACCENT, label="train")
    axes[1].plot(history.epoch, history.val_accuracy, "o-", ms=3, color="#E2663C",
                 label="validation")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].legend(frameon=False, labelcolor=DIM, fontsize=8)

    best = history.val_qwk.max()
    best_epoch = int(history.loc[history.val_qwk.idxmax(), "epoch"])
    axes[2].plot(history.epoch, history.val_qwk, "o-", ms=3, color="#3FB98A")
    axes[2].axhline(best, ls="--", lw=0.8, color=FAINT)
    axes[2].plot([best_epoch], [best], "o", ms=8, mfc="none", mec="#3FB98A", mew=1.6)
    axes[2].annotate(f"best {best:.3f}\nepoch {best_epoch}", (best_epoch, best),
                     textcoords="offset points", xytext=(-6, -32),
                     fontsize=7.5, color="#3FB98A", ha="center")
    axes[2].set_title("Agreement with the human grader (QWK)")
    axes[2].set_xlabel("epoch")

    fig.suptitle("Training history: 20 of 25 epochs, stopped early when agreement plateaued",
                 fontsize=11.5, y=1.02)
    fig.text(0.5, -0.06,
             "Training accuracy is measured on augmented, class-balanced batches, so it is not "
             "directly comparable with validation accuracy. The widening loss gap after epoch 12 "
             "is mild overfitting; early stopping on QWK is what caps it.",
             ha="center", fontsize=7.6, color=FAINT)
    fig.tight_layout()
    save(fig, "fig_training_curves")


def figure_internal_vs_external(evaluation: dict) -> None:
    internal = evaluation["ddr_test"]
    external = evaluation["aptos_external"]

    metrics = [
        ("Accuracy", internal["accuracy"], external["accuracy"]),
        ("Macro precision", internal["macro_precision"], external["macro_precision"]),
        ("Macro recall", internal["macro_recall"], external["macro_recall"]),
        ("Macro F1", internal["macro_f1"], external["macro_f1"]),
        ("Weighted F1", internal["weighted_f1"], external["weighted_f1"]),
        ("QWK", internal["qwk"], external["qwk"]),
        ("Referable AUC", internal["referable"]["auc"], external["referable"]["auc"]),
    ]

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    positions = np.arange(len(metrics))
    height = 0.36

    internal_values = [row[1] for row in metrics]
    external_values = [row[2] for row in metrics]

    ax.barh(positions + height / 2, internal_values, height,
            label="DDR test — same hospitals as training", color=ACCENT, edgecolor="none")
    ax.barh(positions - height / 2, external_values, height,
            label="APTOS 2019 — never trained on", color="#CF3D57", edgecolor="none")

    for position, (_label, inside, outside) in zip(positions, metrics, strict=True):
        ax.text(inside + 0.012, position + height / 2, f"{inside:.3f}",
                va="center", fontsize=7.5, color=DIM)
        ax.text(outside + 0.012, position - height / 2, f"{outside:.3f}",
                va="center", fontsize=7.5, color=DIM)
        gap = inside - outside
        ax.text(1.26, position, f"−{gap:.3f}", va="center", ha="right",
                fontsize=7.5, color="#E9A13B" if gap > 0.1 else FAINT)

    ax.set_yticks(positions, [row[0] for row in metrics], fontsize=8.5)
    ax.set_xlim(0, 1.28)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_xlabel("score")
    ax.set_title("What happens on a different country, camera and population")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.42, -0.14),
              ncol=2, labelcolor=DIM, fontsize=8)
    ax.axvline(1.13, color=LINE, lw=0.8)
    ax.text(1.26, len(metrics) - 0.42, "gap", ha="right", fontsize=7.5, color=FAINT)

    fig.text(0.5, -0.13,
             "Macro F1 falls 0.27 but referable-DR AUC falls only 0.017. The model still "
             "separates sick from healthy on new data; what it loses is the ability to name\n"
             "the exact stage. For a screening tool that triages rather than diagnoses, "
             "that is the better failure mode.",
             ha="center", fontsize=7.8, color=DIM)
    fig.tight_layout()
    save(fig, "fig_internal_vs_external")


def figure_risk_coverage() -> None:
    sweep = pd.read_csv(ARTIFACTS / "tables" / "table9_referral_sweep.csv")

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.plot(sweep["deferred_%"], sweep["accuracy_on_kept"], "o-",
            color=ACCENT, label="stage accuracy on the cases kept")
    ax.plot(sweep["deferred_%"], sweep["referable_sensitivity_on_kept"], "s-",
            color="#3FB98A", label="referable-DR sensitivity on the cases kept")

    for _, row in sweep.iterrows():
        ax.text(row["deferred_%"], row["accuracy_on_kept"] - 0.016,
                f"{row['accuracy_on_kept']:.3f}", ha="center", fontsize=7, color=FAINT)

    ax.set_xlabel("% of cases handed to a human grader")
    ax.set_ylabel("score on the cases the model keeps")
    ax.set_ylim(0.85, 1.0)
    ax.set_title("Does the model's uncertainty actually find its own mistakes?")
    ax.legend(frameon=False, loc="lower right", labelcolor=DIM, fontsize=8)

    fig.text(0.5, -0.04,
             "Yes. Deferring the least certain 20% lifts accuracy from 0.866 to 0.942 and "
             "referable sensitivity from 0.899 to 0.964 — so the deferred cases really were "
             "the hard ones, and deferring does not quietly drop sick patients.",
             ha="center", fontsize=7.8, color=DIM, wrap=True)
    fig.tight_layout()
    save(fig, "fig_risk_coverage")


def figure_triage_distribution() -> None:
    table = pd.read_csv(ARTIFACTS / "tables" / "table9b_triage_actions.csv", index_col=0)
    shares = table.div(table.sum(axis=1), axis=0)

    columns = ["Routine rescreen", "Refer for human grading", "Refer to ophthalmology"]
    colours = [GRADE_COLOURS[0], "#E9A13B", "#CF3D57"]

    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    ax.grid(False)
    left = np.zeros(len(shares))
    positions = np.arange(len(shares))

    for column, colour in zip(columns, colours, strict=True):
        values = shares[column].values
        ax.barh(positions, values, left=left, color=colour, edgecolor=INK,
                linewidth=0.8, label=column)
        for position, (value, offset) in enumerate(zip(values, left, strict=True)):
            if value > 0.05:
                ax.text(offset + value / 2, position,
                        f"{table[column].values[position]}",
                        ha="center", va="center", fontsize=8,
                        color=INK, weight="semibold")
        left += values

    ax.set_yticks(positions, shares.index, fontsize=8.5)
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of images with that true stage")
    ax.set_title("What the triage policy does to each true stage\nDDR test set, 1,879 images")
    ax.legend(frameon=False, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.16), labelcolor=DIM, fontsize=8)
    ax.invert_yaxis()

    fig.text(0.5, -0.16,
             "Every severe case and 134 of 137 proliferative cases are referred. The cost is "
             "36 healthy eyes sent on unnecessarily — the right direction to err for screening.",
             ha="center", fontsize=7.8, color=DIM)
    fig.tight_layout()
    save(fig, "fig_triage_distribution")


def figure_class_imbalance() -> None:
    distribution = pd.read_csv(ARTIFACTS / "tables" / "table1_class_distribution.csv")
    sampler = pd.read_csv(ARTIFACTS / "tables" / "table4_sampler_effect.csv")

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(11, 4.2))

    bars = ax_left.bar(GRADE_SHORT, distribution["count"], color=GRADE_COLOURS,
                       edgecolor="none")
    ax_left.set_yscale("log")
    ax_left.set_ylabel("images (log scale)")
    ax_left.set_title("DDR is heavily imbalanced")
    for bar, count, percent in zip(bars, distribution["count"], distribution["percent"], strict=True):
        ax_left.text(bar.get_x() + bar.get_width() / 2, count * 1.12,
                     f"{count}\n{percent}%", ha="center", fontsize=7.5, color=DIM)
    ax_left.set_ylim(100, 20000)

    positions = np.arange(5)
    width = 0.38
    ax_right.bar(positions - width / 2, sampler["original_%"], width,
                 label="as collected", color="#3A4F6B", edgecolor="none")
    ax_right.bar(positions + width / 2, sampler["after_sampling_%"], width,
                 label="after balanced sampling", color=ACCENT, edgecolor="none")
    ax_right.set_xticks(positions, GRADE_SHORT, fontsize=8)
    ax_right.set_ylabel("% of a training epoch")
    ax_right.set_title("Effective-number weighting rebalances the batches")
    ax_right.legend(frameon=False, labelcolor=DIM, fontsize=8)

    fig.text(0.5, -0.03,
             "Half the dataset is healthy and only 1.9% is severe — a 26.6 : 1 ratio. A model "
             "that answered \"No DR\" every time would score 50% accuracy, which is why macro "
             "F1 and QWK are the metrics that matter here.",
             ha="center", fontsize=7.8, color=DIM)
    fig.tight_layout()
    save(fig, "fig_class_imbalance")


def figure_cpu_latency(evaluation: dict) -> None:
    benchmarks = pd.DataFrame(evaluation["deployment"]["benchmarks"])

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(9.5, 3.8))
    names = [name.replace("backbone_", "").replace(".onnx", "") for name in benchmarks["model"]]

    bars = ax_left.bar(names, benchmarks["size_mb"], color=[ACCENT, "#3FB98A"],
                       edgecolor="none", width=0.5)
    for bar, value in zip(bars, benchmarks["size_mb"], strict=True):
        ax_left.text(bar.get_x() + bar.get_width() / 2, value + 2, f"{value} MB",
                     ha="center", fontsize=8, color=DIM)
    ax_left.set_ylabel("file size")
    ax_left.set_title("Export size")
    ax_left.set_ylim(0, 100)

    bars = ax_right.bar(names, benchmarks["median_ms"], color=[ACCENT, "#3FB98A"],
                        edgecolor="none", width=0.5)
    for bar, value in zip(bars, benchmarks["median_ms"], strict=True):
        ax_right.text(bar.get_x() + bar.get_width() / 2, value + 8, f"{value:.0f} ms",
                      ha="center", fontsize=8, color=DIM)
    ax_right.set_ylabel("median latency per image")
    ax_right.set_title("CPU speed")
    ax_right.set_ylim(0, 420)

    fig.suptitle("Why the app ships fp32 and not the smaller int8 export", fontsize=11)
    fig.text(0.5, -0.06,
             "Dynamic int8 quantisation cut the file to a quarter but made inference 1.8× "
             "slower: the per-operator conversion overhead is not repaid by a convolutional "
             "backbone. Measured, not assumed.",
             ha="center", fontsize=7.8, color=DIM)
    fig.tight_layout()
    save(fig, "fig_cpu_latency")


# ===========================================================================


def main() -> None:
    evaluation = load_evaluation()
    print("Writing figures to docs/images/\n")

    print(" architecture")
    figure_system_architecture()
    figure_model_architecture()
    figure_preprocessing_pipeline()
    figure_triage_flow()

    print(" evaluation")
    figure_confusion_matrix(evaluation)
    figure_per_class_metrics(evaluation)
    figure_training_curves()
    figure_internal_vs_external(evaluation)
    figure_risk_coverage()
    figure_triage_distribution()
    figure_class_imbalance()
    figure_cpu_latency(evaluation)

    print(f"\ndone — {len(list(OUTPUT.glob('*.png')))} figures in {OUTPUT}")


if __name__ == "__main__":
    main()
