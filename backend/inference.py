
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .config import GRADE_NAMES, Settings



def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def corn_cumulative_probs(logits: np.ndarray) -> np.ndarray:
    """P(grade > k) for k = 0..3, guaranteed non-increasing across k."""
    return np.cumprod(sigmoid(logits), axis=1)


def cumulative_to_grade_probs(cumulative: np.ndarray) -> np.ndarray:
    """P(grade > k) -> P(grade == k) for the five stages."""
    ones = np.ones((cumulative.shape[0], 1), dtype=cumulative.dtype)
    zeros = np.zeros((cumulative.shape[0], 1), dtype=cumulative.dtype)
    padded = np.concatenate([ones, cumulative, zeros], axis=1)
    return np.clip(padded[:, :-1] - padded[:, 1:], 0.0, None)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = x - x.max(axis=axis, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=axis, keepdims=True)



@dataclass
class Prediction:
    """One image's full readout."""

    grade: int
    grade_name: str
    expected_grade: float          # continuous severity, 0.0 - 4.0
    grade_probabilities: list[float]
    confidence: float              # probability of the winning stage
    uncertainty: float             # spread of severity across MC samples
    entropy: float
    probability_ungradable: float
    probability_any_dr: float      # P(grade >= 1)
    probability_referable: float   # P(grade >= 2)
    mc_samples: int
    inference_ms: float

    def to_dict(self) -> dict:
        return {
            "grade": self.grade,
            "grade_name": self.grade_name,
            "expected_grade": round(self.expected_grade, 4),
            "grade_probabilities": [round(p, 5) for p in self.grade_probabilities],
            "confidence": round(self.confidence, 4),
            "uncertainty": round(self.uncertainty, 4),
            "entropy": round(self.entropy, 4),
            "probability_ungradable": round(self.probability_ungradable, 4),
            "probability_any_dr": round(self.probability_any_dr, 4),
            "probability_referable": round(self.probability_referable, 4),
            "mc_samples": self.mc_samples,
            "inference_ms": round(self.inference_ms, 1),
        }


class ModelNotLoaded(RuntimeError):
    """Raised when a prediction is requested but no weights are on disk."""


class DRTriageEngine:
    """Loads the exported model once and serves predictions from it."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.session = None
        self.cam_session = None
        self.input_name: str | None = None
        self.cam_input_name: str | None = None
        self.heads: dict[str, np.ndarray] = {}
        self.dropout_p: float = 0.4
        self.load_error: str | None = None
        self.feature_size: int | None = None
        self._rng = np.random.default_rng(42)

    # -- loading ------------------------------------------------------------

    def load(self) -> bool:
        """Load the ONNX backbone and the head weights. Returns True on success."""
        backbone_path = self.settings.backbone_path
        heads_path = self.settings.heads_path

        missing = [str(p) for p in (backbone_path, heads_path) if not p.exists()]
        if missing:
            self.load_error = (
                "Model files not found: " + ", ".join(missing) + ". "
                "Run scripts/download_models.py, or export them with "
                "scripts/export_deployment.py."
            )
            return False

        try:
            import onnxruntime as ort
        except ImportError as error:  # pragma: no cover - dependency guard
            self.load_error = f"onnxruntime is not installed: {error}"
            return False

        options = ort.SessionOptions()
        options.intra_op_num_threads = self.settings.onnx_threads
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        try:
            self.session = ort.InferenceSession(
                str(backbone_path), options, providers=["CPUExecutionProvider"]
            )
            self.input_name = self.session.get_inputs()[0].name

            weights = np.load(heads_path)
            self.heads = {
                "grade_w": weights["grade_w"].astype(np.float32),
                "grade_b": weights["grade_b"].astype(np.float32),
                "quality_w": weights["quality_w"].astype(np.float32),
                "quality_b": weights["quality_b"].astype(np.float32),
            }
            self.dropout_p = float(weights["dropout_p"])
            self.feature_size = int(self.heads["grade_w"].shape[1])
        except Exception as error:  # pragma: no cover - surfaced in /api/health
            self.load_error = f"Failed to load model: {error}"
            self.session = None
            return False

        # The CAM backbone is optional. Without it the app still grades images,
        # it just cannot draw a heatmap.
        cam_path = self.settings.cam_backbone_path
        if self.settings.enable_cam and cam_path.exists():
            try:
                self.cam_session = ort.InferenceSession(
                    str(cam_path), options, providers=["CPUExecutionProvider"]
                )
                self.cam_input_name = self.cam_session.get_inputs()[0].name
            except Exception:
                self.cam_session = None

        self.load_error = None
        return True

    @property
    def is_ready(self) -> bool:
        return self.session is not None

    @property
    def explains(self) -> bool:
        return self.cam_session is not None

    # -- forward passes -----------------------------------------------------

    def extract_features(self, tensor: np.ndarray) -> np.ndarray:
        """NCHW float32 batch -> (batch, feature_size) pooled features."""
        if self.session is None:
            raise ModelNotLoaded(self.load_error or "model not loaded")
        outputs = self.session.run(None, {self.input_name: tensor.astype(np.float32)})
        return np.asarray(outputs[0], dtype=np.float32)

    def extract_feature_map(self, tensor: np.ndarray) -> np.ndarray | None:
        """NCHW batch -> (channels, height, width) final conv map, or None."""
        if self.cam_session is None:
            return None
        outputs = self.cam_session.run(
            None, {self.cam_input_name: tensor.astype(np.float32)}
        )
        # The CAM export returns (pooled_features, feature_map); take the 4-D one.
        for output in outputs:
            array = np.asarray(output)
            if array.ndim == 4:
                return array[0].astype(np.float32)
        return None

    def _mc_dropout_heads(self, features: np.ndarray, n_samples: int):
        """
        Repeat the dropout-and-heads step n_samples times.

        This reproduces PyTorch's inverted dropout: the surviving activations are
        scaled by 1/(1-p) so the expected value is unchanged, which is what the
        model saw during training.
        """
        keep_probability = 1.0 - self.dropout_p
        cumulative_samples = []
        severity_samples = []
        quality_samples = []

        for _ in range(n_samples):
            mask = (self._rng.random(features.shape) < keep_probability).astype(np.float32)
            dropped = features * mask / max(keep_probability, 1e-8)

            grade_logits = dropped @ self.heads["grade_w"].T + self.heads["grade_b"]
            quality_logits = dropped @ self.heads["quality_w"].T + self.heads["quality_b"]

            cumulative = corn_cumulative_probs(grade_logits)
            cumulative_samples.append(cumulative)
            severity_samples.append(cumulative.sum(axis=1))
            quality_samples.append(softmax(quality_logits, axis=1))

        return (
            np.stack(cumulative_samples).mean(axis=0),
            np.stack(severity_samples),
            np.stack(quality_samples).mean(axis=0),
        )

    def predict(self, tensor: np.ndarray, n_samples: int | None = None) -> Prediction:
        """Full readout for a single preprocessed image."""
        n_samples = n_samples or self.settings.mc_dropout_samples
        started = time.perf_counter()

        features = self.extract_features(tensor)
        mean_cumulative, severity, quality_probs = self._mc_dropout_heads(features, n_samples)

        grade_probs = cumulative_to_grade_probs(mean_cumulative)[0]
        total = grade_probs.sum()
        if total > 0:
            grade_probs = grade_probs / total

        grade = int((mean_cumulative[0] > 0.5).sum())
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        return Prediction(
            grade=grade,
            grade_name=GRADE_NAMES[grade],
            expected_grade=float(severity.mean(axis=0)[0]),
            grade_probabilities=grade_probs.tolist(),
            confidence=float(grade_probs.max()),
            uncertainty=float(severity.std(axis=0)[0]),
            entropy=float(-(grade_probs * np.log(grade_probs + 1e-12)).sum()),
            probability_ungradable=float(quality_probs[0][1]),
            probability_any_dr=float(mean_cumulative[0][0]),
            probability_referable=float(mean_cumulative[0][1]),
            mc_samples=n_samples,
            inference_ms=elapsed_ms,
        )

    # -- diagnostics --------------------------------------------------------

    def describe(self) -> dict:
        return {
            "ready": self.is_ready,
            "explainability": self.explains,
            "backbone": str(self.settings.backbone_path),
            "backbone_size_mb": (
                round(self.settings.backbone_path.stat().st_size / 1e6, 1)
                if self.settings.backbone_path.exists()
                else None
            ),
            "feature_size": self.feature_size,
            "dropout_p": self.dropout_p,
            "mc_dropout_samples": self.settings.mc_dropout_samples,
            "error": self.load_error,
        }
