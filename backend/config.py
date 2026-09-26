"""
Application settings.

Every threshold here is copied from the `Config` dataclass in the training
notebook. They are read from environment variables so a deployment can be
retuned (for example, a stricter referral threshold for a high-risk clinic)
without editing code.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

GRADE_NAMES = {
    0: "No DR",
    1: "Mild NPDR",
    2: "Moderate NPDR",
    3: "Severe NPDR",
    4: "Proliferative DR",
}

GRADE_SHORT = ["No DR", "Mild", "Moderate", "Severe", "PDR"]

GRADE_DESCRIPTIONS = {
    0: "No visible signs of diabetic retinopathy.",
    1: "Microaneurysms only. Non-proliferative, earliest detectable stage.",
    2: "More than microaneurysms but less than severe. Referral threshold.",
    3: "Extensive haemorrhages, venous beading or IRMA. High risk of progression.",
    4: "Neovascularisation or vitreous haemorrhage. Sight-threatening.",
}

TRIAGE_ACTIONS = {
    "recapture": "Re-capture image",
    "human": "Refer for human grading",
    "ophthalmology": "Refer to ophthalmology",
    "routine": "Routine rescreen",
}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    # --- image pipeline -------------------------------------------------
    image_size: int = field(default_factory=lambda: _env_int("DR_IMAGE_SIZE", 384))
    use_graham: bool = field(default_factory=lambda: _env_bool("DR_USE_GRAHAM", True))
    use_clahe: bool = field(default_factory=lambda: _env_bool("DR_USE_CLAHE", True))

    # --- model files ----------------------------------------------------
    model_dir: Path = field(
        default_factory=lambda: Path(os.environ.get("DR_MODEL_DIR", PROJECT_ROOT / "models"))
    )
    backbone_file: str = field(
        default_factory=lambda: os.environ.get("DR_BACKBONE_FILE", "backbone_fp32.onnx")
    )
    heads_file: str = field(default_factory=lambda: os.environ.get("DR_HEADS_FILE", "heads.npz"))
    cam_backbone_file: str = field(
        default_factory=lambda: os.environ.get("DR_CAM_FILE", "backbone_cam.onnx")
    )

    # --- inference ------------------------------------------------------
    num_classes: int = 5
    mc_dropout_samples: int = field(default_factory=lambda: _env_int("DR_MC_SAMPLES", 20))
    onnx_threads: int = field(default_factory=lambda: _env_int("DR_ONNX_THREADS", 2))

    # --- triage thresholds (from the notebook Config) -------------------
    ungradable_threshold: float = field(
        default_factory=lambda: _env_float("DR_UNGRADABLE_THRESHOLD", 0.50)
    )
    uncertainty_threshold: float = field(
        default_factory=lambda: _env_float("DR_UNCERTAINTY_THRESHOLD", 0.45)
    )
    confidence_threshold: float = field(
        default_factory=lambda: _env_float("DR_CONFIDENCE_THRESHOLD", 0.55)
    )
    referral_grade: int = field(default_factory=lambda: _env_int("DR_REFERRAL_GRADE", 2))

    # --- serving --------------------------------------------------------
    max_upload_mb: int = field(default_factory=lambda: _env_int("DR_MAX_UPLOAD_MB", 12))
    allow_demo_mode: bool = field(default_factory=lambda: _env_bool("DR_ALLOW_DEMO", True))
    artifacts_dir: Path = field(
        default_factory=lambda: Path(
            os.environ.get("DR_ARTIFACTS_DIR", PROJECT_ROOT / "artifacts")
        )
    )

    @property
    def backbone_path(self) -> Path:
        return self.model_dir / self.backbone_file

    @property
    def heads_path(self) -> Path:
        return self.model_dir / self.heads_file

    @property
    def cam_backbone_path(self) -> Path:
        return self.model_dir / self.cam_backbone_file

    def to_dict(self) -> dict:
        data = asdict(self)
        for key, value in data.items():
            if isinstance(value, Path):
                data[key] = str(value)
        return data


settings = Settings()
