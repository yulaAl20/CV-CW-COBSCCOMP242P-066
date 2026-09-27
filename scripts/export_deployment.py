
from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

try:
    import timm
except ImportError as error:  # pragma: no cover
    raise SystemExit("timm is required to rebuild the model. pip install timm") from error


DEFAULT_BACKBONE = "tf_efficientnetv2_s.in21k_ft_in1k"


class DRTriageNet(nn.Module):
    def __init__(self, backbone_name, num_classes=5, num_quality_classes=2,
                 dropout=0.4, pretrained=False):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        feature_size = self.backbone.num_features
        self.dropout = nn.Dropout(p=dropout)
        self.grade_head = nn.Linear(feature_size, num_classes - 1)
        self.quality_head = nn.Linear(feature_size, num_quality_classes)

    def forward_features(self, images):
        return self.backbone(images)

    def forward(self, images):
        dropped = self.dropout(self.backbone(images))
        return self.grade_head(dropped), self.quality_head(dropped)


class BackboneOnly(nn.Module):
    """The ONNX graph for the fast path: image in, pooled features out."""

    def __init__(self, network):
        super().__init__()
        self.network = network

    def forward(self, images):
        return self.network.forward_features(images)


class BackboneWithMap(nn.Module):
    """
    The ONNX graph for explanations: pooled features plus the conv map.

    timm's EfficientNetV2 exposes `forward_features` on the backbone itself,
    which returns the spatial map before pooling. Pooling it here reproduces
    exactly what the fast path returns, so the two graphs stay consistent.
    """

    def __init__(self, network):
        super().__init__()
        self.network = network

    def forward(self, images):
        feature_map = self.network.backbone.forward_features(images)
        pooled = feature_map.mean(dim=(2, 3))
        return pooled, feature_map


# ---------------------------------------------------------------------------


def load_checkpoint(path: Path, backbone_name: str, dropout: float, prefer_ema: bool):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    if prefer_ema and checkpoint.get("ema"):
        weights = checkpoint["ema"]
        source = "ema"
    else:
        weights = checkpoint.get("model", checkpoint)
        source = "model"

    settings = checkpoint.get("config", {})
    backbone_name = settings.get("backbone", backbone_name)
    dropout = settings.get("dropout", dropout)
    image_size = settings.get("image_size", 384)

    model = DRTriageNet(backbone_name, dropout=dropout, pretrained=False)
    model.load_state_dict({name: tensor.cpu() for name, tensor in weights.items()})
    model.eval()

    print(f"[checkpoint] {path.name}")
    print(f"  weights   : {source}")
    print(f"  backbone  : {backbone_name}")
    print(f"  epoch     : {checkpoint.get('epoch', 'unknown')}")
    print(f"  best QWK  : {checkpoint.get('best_qwk', 'unknown')}")
    return model, backbone_name, int(image_size)


def export_graph(module: nn.Module, path: Path, image_size: int,
                 output_names: list[str], opset: int = 17) -> None:
    extra = {}
    # PyTorch 2.9 switched the default exporter; pin the TorchScript one so the
    # graph shape stays what onnxruntime expects.
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        extra["dynamo"] = False

    dynamic_axes = {"image": {0: "batch"}}
    for name in output_names:
        dynamic_axes[name] = {0: "batch"}

    torch.onnx.export(
        module,
        torch.randn(1, 3, image_size, image_size),
        str(path),
        input_names=["image"],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        do_constant_folding=True,
        **extra,
    )
    print(f"[onnx] {path.name}  ({path.stat().st_size / 1e6:.1f} MB)")


def export_heads(model: DRTriageNet, path: Path) -> None:
    np.savez(
        path,
        grade_w=model.grade_head.weight.detach().cpu().numpy(),
        grade_b=model.grade_head.bias.detach().cpu().numpy(),
        quality_w=model.quality_head.weight.detach().cpu().numpy(),
        quality_b=model.quality_head.bias.detach().cpu().numpy(),
        dropout_p=np.array(model.dropout.p, dtype=np.float32),
    )
    print(f"[heads] {path.name}  ({path.stat().st_size / 1e3:.0f} KB)")


def verify(model: DRTriageNet, onnx_path: Path, image_size: int,
           tolerance: float = 1e-3) -> dict:
    """Compare ONNX output against PyTorch on a random input."""
    import onnxruntime as ort

    probe = torch.randn(1, 3, image_size, image_size)
    with torch.no_grad():
        reference = model.forward_features(probe).numpy()

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    produced = session.run(None, {session.get_inputs()[0].name: probe.numpy()})[0]

    difference = float(np.abs(reference - produced).max())
    passed = difference < tolerance
    print(f"[verify] max difference {difference:.2e}  (tolerance {tolerance})  "
          f"-> {'PASS' if passed else 'FAIL'}")
    return {"verified": passed, "max_abs_diff": difference, "tolerance": tolerance}


def quantize(source: Path, target: Path) -> dict | None:
    """
    Produce an int8 copy.

    Worth measuring, not worth assuming: on the T4 export the int8 graph came
    out roughly four times smaller but almost twice as slow on CPU, because
    dynamic quantisation adds per-operator conversion overhead that a
    convolutional backbone does not pay back. Keep fp32 as the default and only
    ship int8 if disk is the binding constraint.
    """
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        quantize_dynamic(str(source), str(target), weight_type=QuantType.QUInt8)
    except Exception as error:
        print(f"[int8] skipped: {error}")
        return None

    before = source.stat().st_size / 1e6
    after = target.stat().st_size / 1e6
    print(f"[int8] {before:.1f} MB -> {after:.1f} MB")
    return {"size_mb": round(after, 1)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--out", default=Path("models"), type=Path)
    parser.add_argument("--backbone", default=DEFAULT_BACKBONE)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--no-ema", action="store_true",
                        help="export the raw weights instead of the averaged ones")
    parser.add_argument("--skip-cam", action="store_true",
                        help="do not export the heatmap graph")
    parser.add_argument("--int8", action="store_true",
                        help="also write a dynamically quantised copy")
    arguments = parser.parse_args()

    arguments.out.mkdir(parents=True, exist_ok=True)

    model, backbone_name, image_size = load_checkpoint(
        arguments.checkpoint, arguments.backbone, arguments.dropout, not arguments.no_ema
    )
    if arguments.image_size:
        image_size = arguments.image_size

    fast_path = arguments.out / "backbone_fp32.onnx"
    export_graph(BackboneOnly(model), fast_path, image_size, ["features"])
    export_heads(model, arguments.out / "heads.npz")

    if not arguments.skip_cam:
        export_graph(
            BackboneWithMap(model),
            arguments.out / "backbone_cam.onnx",
            image_size,
            ["features", "feature_map"],
        )

    manifest = {
        "backbone": backbone_name,
        "image_size": image_size,
        "feature_size": int(model.grade_head.in_features),
        "dropout_p": float(model.dropout.p),
        "weights": "raw" if arguments.no_ema else "ema",
        "verification": verify(model, fast_path, image_size),
    }

    if arguments.int8:
        manifest["int8"] = quantize(fast_path, arguments.out / "backbone_int8.onnx")

    (arguments.out / "export_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nexport complete -> {arguments.out.resolve()}")

    if not manifest["verification"]["verified"]:
        raise SystemExit("Export verification FAILED. Do not deploy these files.")


if __name__ == "__main__":
    main()
