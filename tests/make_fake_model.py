
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

FEATURE_SIZE = 1280
IMAGE_SIZE = 384


def build_backbone(feature_size: int = FEATURE_SIZE, with_map: bool = False) -> onnx.ModelProto:
    """image (N,3,H,W) -> features (N,feature_size), optionally also the conv map."""
    rng = np.random.default_rng(0)

    # A 1x1 convolution to 'feature_size' channels stands in for the whole
    # backbone: it produces a spatial map, which is then average pooled.
    kernel = numpy_helper.from_array(
        (rng.normal(0, 0.05, (feature_size, 3, 1, 1))).astype(np.float32), "kernel"
    )

    nodes = [
        helper.make_node("Conv", ["image", "kernel"], ["conv"], kernel_shape=[1, 1]),
        helper.make_node("Relu", ["conv"], ["feature_map"]),
        helper.make_node("GlobalAveragePool", ["feature_map"], ["pooled"]),
        helper.make_node("Flatten", ["pooled"], ["features"], axis=1),
    ]

    inputs = [helper.make_tensor_value_info("image", TensorProto.FLOAT, ["N", 3, "H", "W"])]
    outputs = [helper.make_tensor_value_info("features", TensorProto.FLOAT, ["N", feature_size])]
    if with_map:
        outputs.append(
            helper.make_tensor_value_info(
                "feature_map", TensorProto.FLOAT, ["N", feature_size, "h", "w"]
            )
        )

    graph = helper.make_graph(nodes, "fake_backbone", inputs, outputs, initializer=[kernel])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    return model


def build_heads(feature_size: int = FEATURE_SIZE) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(1)
    return {
        "grade_w": rng.normal(0, 0.01, (4, feature_size)).astype(np.float32),
        "grade_b": np.zeros(4, dtype=np.float32),
        "quality_w": rng.normal(0, 0.01, (2, feature_size)).astype(np.float32),
        "quality_b": np.array([1.5, -1.5], dtype=np.float32),
        "dropout_p": np.array(0.4, dtype=np.float32),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="models", help="output directory")
    arguments = parser.parse_args()

    out = Path(arguments.out)
    out.mkdir(parents=True, exist_ok=True)

    onnx.save(build_backbone(), out / "backbone_fp32.onnx")
    onnx.save(build_backbone(with_map=True), out / "backbone_cam.onnx")
    np.savez(out / "heads.npz", **build_heads())

    print(f"stand-in model written to {out.resolve()}")
    print("  backbone_fp32.onnx, backbone_cam.onnx, heads.npz")
    print("  These are NOT trained weights. Replace them before reporting any result.")


if __name__ == "__main__":
    main()
