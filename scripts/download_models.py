
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Replace with your own release once you have published one.
DEFAULT_BASE_URL = os.environ.get(
    "DR_WEIGHTS_URL",
    "https://github.com/yulaAl20/CV-CW-COBSCCOMP242P-066/releases/download/v1.0.0",
)

REQUIRED = ["backbone_fp32.onnx", "heads.npz"]
OPTIONAL = ["backbone_cam.onnx", "export_manifest.json"]


def human(size: float) -> str:
    return f"{size / 1e6:.1f} MB" if size >= 1e6 else f"{size / 1e3:.0f} KB"


def download(url: str, target: Path) -> bool:
    print(f"  {target.name:24s} ", end="", flush=True)

    partial = target.with_suffix(target.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            total = int(response.headers.get("Content-Length", 0))
            written = 0
            digest = hashlib.sha256()

            with partial.open("wb") as file:
                while chunk := response.read(1 << 16):
                    file.write(chunk)
                    digest.update(chunk)
                    written += len(chunk)
                    if total:
                        done = int(28 * written / total)
                        print(f"\r  {target.name:24s} [{'=' * done}{' ' * (28 - done)}] "
                              f"{human(written)}", end="", flush=True)

        partial.replace(target)
        print(f"\r  {target.name:24s} {human(written)}  sha256 {digest.hexdigest()[:12]}"
              + " " * 20)
        return True

    except urllib.error.HTTPError as error:
        partial.unlink(missing_ok=True)
        print(f"\r  {target.name:24s} HTTP {error.code}" + " " * 30)
        return False
    except Exception as error:
        partial.unlink(missing_ok=True)
        print(f"\r  {target.name:24s} failed: {error}" + " " * 20)
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "models")
    parser.add_argument("--include-cam", action="store_true",
                        help="also fetch the heatmap graph (another ~81 MB)")
    parser.add_argument("--force", action="store_true", help="re-download existing files")
    arguments = parser.parse_args()

    if "YOUR-USERNAME" in arguments.base_url:
        print("No weights URL is configured yet.\n")
        print("Either publish a GitHub Release with the exported files attached and")
        print("edit DEFAULT_BASE_URL in this script, or export them yourself:\n")
        print("    python scripts/export_deployment.py --checkpoint best.pt --out models\n")
        print("The app runs in demo mode until the weights are in place.")
        sys.exit(1)

    arguments.out.mkdir(parents=True, exist_ok=True)
    wanted = REQUIRED + (OPTIONAL if arguments.include_cam else ["export_manifest.json"])

    print(f"Fetching weights from {arguments.base_url}\n")
    failures = []
    for name in wanted:
        target = arguments.out / name
        if target.exists() and not arguments.force:
            print(f"  {name:24s} already present, skipping")
            continue
        if not download(f"{arguments.base_url.rstrip('/')}/{name}", target):
            if name in REQUIRED:
                failures.append(name)

    print()
    if failures:
        print(f"Could not fetch: {', '.join(failures)}")
        print("The app will start in demo mode. Check the release URL and try again.")
        sys.exit(1)

    print(f"Weights ready in {arguments.out.resolve()}")
    print("Start the app with:  python app.py")


if __name__ == "__main__":
    main()
