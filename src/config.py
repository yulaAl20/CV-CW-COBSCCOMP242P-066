
from __future__ import annotations

from dataclasses import dataclass, asdict, field


@dataclass
class Config:
    # ---- data -----------------------------------------------------------
    ddr_root: str = "/kaggle/input/ddr-dataset/DDR-dataset"
    aptos_root: str = "/kaggle/input/aptos2019-blindness-detection"
    image_size: int = 384          # fine-tuning resolution
    warmup_image_size: int = 256   # progressive resizing: cheaper first stage
    num_classes: int = 5
    num_quality_classes: int = 2   # gradable / ungradable (DDR class 5)
    val_frac: float = 0.15
    test_frac: float = 0.15
    seed: int = 42

    # ---- preprocessing ablation switches --------------------------------
    use_graham: bool = True
    use_clahe: bool = True

    # ---- model ----------------------------------------------------------
    backbone: str = "tf_efficientnetv2_s.in21k_ft_in1k"
    # Deployment fallback if CPU latency exceeds budget: "tf_efficientnetv2_b0"
    pretrained: bool = True
    dropout: float = 0.4           # also the MC-dropout rate at inference

    # ---- optimisation ---------------------------------------------------
    epochs: int = 25   
    batch_size: int = 32           
    head_lr: float = 1e-3
    backbone_lr: float = 1e-4
    weight_decay: float = 1e-5
    warmup_steps: int = 300
    grad_clip: float = 1.0
    amp: bool = True
    use_ema: bool = True
    ema_decay: float = 0.999
    freeze_first_epoch: bool = True
    unfreeze_last_n: int | None = None   # None == unfreeze everything
    quality_loss_weight: float = 0.3

    # ---- class balancing ------------------------------------------------
    balanced_sampler: bool = True
    sampler_beta: float = 0.999    # effective-number beta (Cui et al., 2019)

    # ---- early stopping -------------------------------------------------
    patience: int = 4
    min_delta: float = 0.002

    # ---- inference ------------------------------------------------------
    mc_dropout_samples: int = 20
    tta: bool = True

    # ---- referral thresholds (tuned on validation, see notebook 04) ------
    ungradable_threshold: float = 0.50
    uncertainty_threshold: float = 0.45
    confidence_threshold: float = 0.55

    # ---- infrastructure -------------------------------------------------
    checkpoint_dir: str = "/kaggle/working/checkpoints"
    export_dir: str = "models/export"
    device: str = "cuda"
    num_workers: int = 2

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT = Config()
