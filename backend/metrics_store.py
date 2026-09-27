
from __future__ import annotations

import csv
import json
from pathlib import Path

from .config import GRADE_NAMES, GRADE_SHORT, Settings


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open() as file:
        return json.load(file)


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def _as_float(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class MetricsStore:
    def __init__(self, settings: Settings):
        self.dir = settings.artifacts_dir
        self._cache: dict | None = None

    def _evaluation(self) -> dict:
        # Read once. The file only changes when a new training run is copied
        # in, which means a restart anyway.
        if self._cache is None:
            self._cache = _read_json(self.dir / "evaluation.json")
        return self._cache

    def headline(self) -> dict:
        """The numbers shown on the performance panel."""
        evaluation = self._evaluation()
        internal = evaluation.get("ddr_test", {})
        external = evaluation.get("aptos_external", {})
        if not internal:
            return {"available": False}

        detection = internal.get("dr_detection", {})
        referable = internal.get("referable", {})

        return {
            "available": True,
            "internal": {
                "dataset": "DDR held-out test",
                "images": internal.get("n_images"),
                "accuracy": internal.get("accuracy"),
                "macro_precision": internal.get("macro_precision"),
                "macro_recall": internal.get("macro_recall"),
                "macro_f1": internal.get("macro_f1"),
                "weighted_f1": internal.get("weighted_f1"),
                "qwk": internal.get("qwk"),
                "referable_auc": referable.get("auc"),
            },
            "external": {
                "dataset": "APTOS 2019 (never trained on)",
                "images": external.get("n_images"),
                "accuracy": external.get("accuracy"),
                "macro_f1": external.get("macro_f1"),
                "qwk": external.get("qwk"),
                "referable_auc": external.get("referable", {}).get("auc"),
            },
            "detection": {
                "accuracy": detection.get("accuracy"),
                "precision": detection.get("precision"),
                "recall": detection.get("recall_sensitivity"),
                "specificity": detection.get("specificity"),
                "f1": detection.get("f1"),
            },
            "calibration": {
                "ece": internal.get("calibration", {}).get("ece"),
                "mce": internal.get("calibration", {}).get("mce"),
                "aurc": internal.get("selective", {}).get("aurc"),
            },
        }

    def per_class(self) -> list[dict]:
        """Precision, recall and F1 for each of the five stages."""
        internal = self._evaluation().get("ddr_test", {})
        precision = internal.get("precision_per_class") or []
        recall = internal.get("recall_per_class") or []
        f1 = internal.get("f1_per_class") or []
        support = internal.get("support_per_class") or []

        rows = []
        for grade in range(5):
            rows.append(
                {
                    "grade": grade,
                    "name": GRADE_NAMES[grade],
                    "short": GRADE_SHORT[grade],
                    "precision": precision[grade] if grade < len(precision) else None,
                    "recall": recall[grade] if grade < len(recall) else None,
                    "f1": f1[grade] if grade < len(f1) else None,
                    "support": support[grade] if grade < len(support) else None,
                }
            )
        return rows

    def confusion_matrix(self) -> dict:
        internal = self._evaluation().get("ddr_test", {})
        matrix = internal.get("confusion_matrix") or []
        return {"labels": GRADE_SHORT, "matrix": matrix}

    def training_history(self) -> list[dict]:
        rows = _read_csv(self.dir / "history.csv")
        return [
            {
                "epoch": int(_as_float(row.get("epoch"), 0)),
                "train_loss": _as_float(row.get("train_loss")),
                "val_loss": _as_float(row.get("val_loss")),
                "train_accuracy": _as_float(row.get("train_accuracy")),
                "val_accuracy": _as_float(row.get("val_accuracy")),
                "val_qwk": _as_float(row.get("val_qwk")),
            }
            for row in rows
        ]

    def referral_sweep(self) -> list[dict]:
        """Accuracy and sensitivity as a function of how much work is deferred."""
        rows = _read_csv(self.dir / "tables" / "table9_referral_sweep.csv")
        return [
            {
                "deferred_percent": _as_float(row.get("deferred_%")),
                "images_kept": _as_float(row.get("images_kept")),
                "accuracy": _as_float(row.get("accuracy_on_kept")),
                "referable_sensitivity": _as_float(row.get("referable_sensitivity_on_kept")),
            }
            for row in rows
        ]

    def comparison(self) -> list[dict]:
        """Internal against external, metric by metric, with the gap."""
        internal = self._evaluation().get("ddr_test", {})
        external = self._evaluation().get("aptos_external", {})
        if not internal or not external:
            return []

        rows = [
            ("Stage accuracy", "accuracy"),
            ("Macro precision", "macro_precision"),
            ("Macro recall", "macro_recall"),
            ("Macro F1", "macro_f1"),
            ("Weighted F1", "weighted_f1"),
            ("Agreement (QWK)", "qwk"),
        ]

        out = []
        for label, key in rows:
            inside, outside = internal.get(key), external.get(key)
            out.append({
                "metric": label,
                "internal": inside,
                "external": outside,
                "gap": round(inside - outside, 4) if None not in (inside, outside) else None,
            })

        inside = internal.get("referable", {}).get("auc")
        outside = external.get("referable", {}).get("auc")
        out.append({
            "metric": "Referable DR AUC",
            "internal": inside,
            "external": outside,
            "gap": round(inside - outside, 4) if None not in (inside, outside) else None,
        })
        return out

    def run_details(self) -> dict:
        """How the model that produced these numbers was trained."""
        return self._evaluation().get("_run", {})

    def deployment(self) -> dict:
        """Export verification and the measured CPU latency."""
        return self._evaluation().get("deployment", {})

    def everything(self) -> dict:
        return {
            "headline": self.headline(),
            "per_class": self.per_class(),
            "confusion": self.confusion_matrix(),
            "history": self.training_history(),
            "referral_sweep": self.referral_sweep(),
            "comparison": self.comparison(),
            "run": self.run_details(),
            "deployment": self.deployment(),
        }
