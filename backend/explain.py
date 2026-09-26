
from __future__ import annotations

import cv2
import numpy as np

from .inference import DRTriageEngine, sigmoid

def severity_weights(engine: DRTriageEngine, feature_vector: np.ndarray) -> np.ndarray:
    
    grade_w = engine.heads["grade_w"]              # (4, feature_size)
    grade_b = engine.heads["grade_b"]              # (4,)

    logits = feature_vector @ grade_w.T + grade_b  # (4,)
    probabilities = sigmoid(logits)
    cumulative = np.cumprod(probabilities)

    # d(expected severity)/d(logit_k), accounting for the cumulative product
    weights = np.zeros_like(logits)
    for k in range(len(logits)):
        # logit k appears in every cumulative term from k onwards
        downstream = cumulative[k:].sum()
        weights[k] = downstream * (1.0 - probabilities[k])

    return (weights[:, None] * grade_w).sum(axis=0)  # (feature_size,)


def class_activation_map(
    engine: DRTriageEngine, tensor: np.ndarray
) -> np.ndarray | None:
    """Return a 0-1 heatmap at the model's input resolution, or None."""
    feature_map = engine.extract_feature_map(tensor)
    if feature_map is None:
        return None

    pooled = feature_map.mean(axis=(1, 2))
    weights = severity_weights(engine, pooled)

    heatmap = np.tensordot(weights, feature_map, axes=([0], [0]))
    heatmap = np.maximum(heatmap, 0.0)

    size = tensor.shape[-1]
    heatmap = cv2.resize(heatmap.astype(np.float32), (size, size), interpolation=cv2.INTER_CUBIC)

    span = heatmap.max() - heatmap.min()
    if span > 1e-8:
        heatmap = (heatmap - heatmap.min()) / span
    else:
        heatmap = np.zeros_like(heatmap)
    return heatmap


def occlusion_map(
    engine: DRTriageEngine,
    tensor: np.ndarray,
    grid: int = 8,
    batch: int = 8,
) -> np.ndarray:
   
    size = tensor.shape[-1]
    step = size // grid

    baseline_features = engine.extract_features(tensor)
    baseline = float(
        np.cumprod(
            sigmoid(baseline_features @ engine.heads["grade_w"].T + engine.heads["grade_b"]),
            axis=1,
        ).sum()
    )

    variants, positions = [], []
    for row in range(grid):
        for column in range(grid):
            occluded = tensor.copy()
            occluded[
                :, :, row * step:(row + 1) * step, column * step:(column + 1) * step
            ] = 0.0
            variants.append(occluded[0])
            positions.append((row, column))

    scores = np.zeros((grid, grid), dtype=np.float32)
    for start in range(0, len(variants), batch):
        chunk = np.stack(variants[start:start + batch])
        features = engine.extract_features(chunk)
        severities = np.cumprod(
            sigmoid(features @ engine.heads["grade_w"].T + engine.heads["grade_b"]), axis=1
        ).sum(axis=1)
        for offset, severity in enumerate(severities):
            row, column = positions[start + offset]
            scores[row, column] = baseline - severity

    scores = np.maximum(scores, 0.0)
    heatmap = cv2.resize(scores, (size, size), interpolation=cv2.INTER_CUBIC)
    span = heatmap.max() - heatmap.min()
    return (heatmap - heatmap.min()) / span if span > 1e-8 else np.zeros_like(heatmap)


def overlay_heatmap(image_bgr: np.ndarray, heatmap: np.ndarray, alpha: float = 0.42) -> np.ndarray:
    """Blend a 0-1 heatmap over a BGR image using the JET colour map."""
    heatmap = cv2.resize(
        heatmap.astype(np.float32), (image_bgr.shape[1], image_bgr.shape[0])
    )
    coloured = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    return cv2.addWeighted(coloured, alpha, image_bgr, 1 - alpha, 0)
