"""
Fundus image preprocessing.

This is a 1:1 port of Section 2 of the training notebook. Serving-time
preprocessing MUST match training-time preprocessing exactly, otherwise the
model sees a different distribution of pixels than it was fitted on and the
reported metrics no longer describe what the app actually does.

Pipeline (in order):
    1. crop to the retina         - removes the black letterbox around the disc
    2. pad to a square            - keeps lesion shapes undistorted
    3. resize to 384 x 384        - the network's fixed input size
    4. CLAHE on the green channel - contrast enhancement where lesions show best
    5. subtract local average     - illumination correction + edge enhancement
    6. circular mask              - removes the bright rim of the field of view
"""

from __future__ import annotations

import cv2
import numpy as np

# ImageNet statistics - the backbone was pretrained with these, so the same
# normalisation has to be applied at inference time.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def retina_mask(image: np.ndarray, threshold: int = 10) -> np.ndarray:
    """Boolean mask of the retinal disc: bright pixels are retina, dark are background."""
    mask = image.max(axis=2) > threshold
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    return mask.astype(bool)


def crop_to_retina(image: np.ndarray, threshold: int = 10) -> np.ndarray:
    """Crop away the black border so the retina fills the frame."""
    mask = retina_mask(image, threshold)
    if not mask.any():
        return image
    rows = np.where(mask.any(axis=1))[0]
    columns = np.where(mask.any(axis=0))[0]
    return image[rows[0]:rows[-1] + 1, columns[0]:columns[-1] + 1]


def pad_to_square(image: np.ndarray) -> np.ndarray:
    """Centre the image on a black square canvas, so resizing does not stretch lesions."""
    height, width = image.shape[:2]
    side = max(height, width)
    top, left = (side - height) // 2, (side - width) // 2
    canvas = np.zeros((side, side, image.shape[2]), dtype=image.dtype)
    canvas[top:top + height, left:left + width] = image
    return canvas


def circular_crop(image: np.ndarray, shrink: float = 0.97) -> np.ndarray:
    """Black out everything outside a centred circle to drop the bright FOV rim."""
    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    radius = int(min(height, width) / 2 * shrink)
    cv2.circle(mask, (width // 2, height // 2), radius, 255, thickness=-1)
    return cv2.bitwise_and(image, image, mask=mask)


def subtract_local_average(
    image: np.ndarray,
    sigma_ratio: float = 30.0,
    weight: float = 4.0,
    bias: int = 128,
) -> np.ndarray:
    """
    Illumination correction and edge enhancement.

    Subtracting a heavily blurred copy of the image removes the slow brightness
    gradient left by the camera flash and amplifies the high-frequency detail
    that microaneurysms and haemorrhages live in.
    """
    sigma = max(image.shape[1] / sigma_ratio, 1.0)
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma)
    return cv2.addWeighted(image, weight, blurred, -weight, bias)


def clahe_green(image: np.ndarray, clip_limit: float = 2.0, tile: int = 8) -> np.ndarray:
    """Contrast-limited adaptive histogram equalisation on the green channel only."""
    enhanced = image.copy()
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
    enhanced[:, :, 1] = clahe.apply(enhanced[:, :, 1])
    return enhanced


def preprocess_fundus(
    image_bgr: np.ndarray,
    size: int = 384,
    use_graham: bool = True,
    use_clahe: bool = True,
) -> np.ndarray:
    """Run the full pipeline. Input and output are both BGR uint8."""
    processed = crop_to_retina(image_bgr)
    processed = pad_to_square(processed)
    processed = cv2.resize(processed, (size, size), interpolation=cv2.INTER_AREA)
    if use_clahe:
        processed = clahe_green(processed)
    if use_graham:
        processed = subtract_local_average(processed)
    return circular_crop(processed)


def preprocessing_stages(
    image_bgr: np.ndarray,
    size: int = 384,
    use_graham: bool = True,
    use_clahe: bool = True,
) -> dict[str, np.ndarray]:
    """
    Same pipeline, but returns every intermediate stage as a BGR image.

    The UI shows these side by side so a reviewer can see what the model is
    actually being shown rather than taking the pipeline on trust.
    """
    cropped = crop_to_retina(image_bgr)
    squared = pad_to_square(cropped)
    resized = cv2.resize(squared, (size, size), interpolation=cv2.INTER_AREA)
    contrast = clahe_green(resized) if use_clahe else resized
    illumination = subtract_local_average(contrast) if use_graham else contrast
    masked = circular_crop(illumination)
    return {
        "raw": cv2.resize(image_bgr, (size, size), interpolation=cv2.INTER_AREA),
        "cropped": cv2.resize(squared, (size, size), interpolation=cv2.INTER_AREA),
        "resized": resized,
        "clahe": contrast,
        "illumination": illumination,
        "final": masked,
    }


def to_model_tensor(image_bgr: np.ndarray, size: int = 384) -> np.ndarray:
    """Preprocessed BGR image -> normalised NCHW float32 batch of one."""
    if image_bgr.shape[:2] != (size, size):
        image_bgr = cv2.resize(image_bgr, (size, size), interpolation=cv2.INTER_AREA)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    normalised = (image_rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(normalised.transpose(2, 0, 1))[None]


def looks_like_fundus(image_bgr: np.ndarray, min_retina_fraction: float = 0.25) -> bool:
   
    mask = retina_mask(image_bgr)
    fraction = float(mask.mean())
    return min_retina_fraction <= fraction <= 0.995
