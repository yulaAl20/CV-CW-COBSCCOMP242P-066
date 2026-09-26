
from __future__ import annotations

import hashlib
import time

import cv2
import numpy as np

from .config import GRADE_NAMES
from .inference import Prediction, cumulative_to_grade_probs


def synthetic_prediction(preprocessed_bgr: np.ndarray, n_samples: int = 20) -> Prediction:
    """A deterministic stand-in readout, derived from the image itself."""
    started = time.perf_counter()

    digest = hashlib.sha256(preprocessed_bgr.tobytes()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))

    # Lesion-like texture energy in the green channel gives the stand-in
    # something image-dependent to respond to, so different uploads differ.
    green = preprocessed_bgr[:, :, 1].astype(np.float32)
    edges = cv2.Laplacian(green, cv2.CV_32F, ksize=3)
    texture = float(np.clip(np.abs(edges).mean() / 12.0, 0.0, 1.0))

    severity = np.clip(texture * 3.4 + rng.normal(0, 0.35), 0.0, 4.0)
    logits = np.array([[severity - k - 0.5 for k in range(4)]], dtype=np.float32) * 2.2
    cumulative = np.cumprod(1.0 / (1.0 + np.exp(-logits)), axis=1)

    grade_probs = cumulative_to_grade_probs(cumulative)[0]
    grade_probs = grade_probs / max(grade_probs.sum(), 1e-8)
    grade = int((cumulative[0] > 0.5).sum())

    return Prediction(
        grade=grade,
        grade_name=GRADE_NAMES[grade],
        expected_grade=float(cumulative.sum()),
        grade_probabilities=grade_probs.tolist(),
        confidence=float(grade_probs.max()),
        uncertainty=float(abs(rng.normal(0.12, 0.06))),
        entropy=float(-(grade_probs * np.log(grade_probs + 1e-12)).sum()),
        probability_ungradable=float(np.clip(0.08 + rng.normal(0, 0.05), 0.0, 1.0)),
        probability_any_dr=float(cumulative[0][0]),
        probability_referable=float(cumulative[0][1]),
        mc_samples=n_samples,
        inference_ms=(time.perf_counter() - started) * 1000.0,
    )
