from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "native" / "hf_diffusion"
BUILD_DIR = SOURCE_DIR / "build"
OUTPUT_DIR = ROOT / "src" / "models" / "high_fidelity"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the optional native HF diffusion extension.")
    parser.add_argument("--clean", action="store_true", help="Delete the build directory before configuring")
    parser.add_argument("--debug", action="store_true", help="Build with Debug config instead of Release")
    args = parser.parse_args()

    try:
        import pybind11
    except ImportError as exc:
        msg = "pybind11 is required. Install dev dependencies with `uv sync --dev` first."
        raise SystemExit(msg) from exc

    if args.clean and BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)
    BUILD_DIR.mkdir(parents=True, exist_ok=True)

    config = "Debug" if args.debug else "Release"
    configure_cmd = [
        "cmake",
        "-S",
        str(SOURCE_DIR),
        "-B",
        str(BUILD_DIR),
        f"-DPython_EXECUTABLE={sys.executable}",
        f"-DPYBIND11_INCLUDE_DIR={pybind11.get_include()}",
        f"-DOUTPUT_DIR={OUTPUT_DIR}",
        f"-DCMAKE_BUILD_TYPE={config}",
    ]
    build_cmd = ["cmake", "--build", str(BUILD_DIR), "--config", config]

    subprocess.run(configure_cmd, check=True)
    subprocess.run(build_cmd, check=True)
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
