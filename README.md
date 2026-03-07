# PITNN

Physics-informed tire thermal simulation project with a stable legacy model and a higher-fidelity simulation path.

## Current Status
- Implemented and tested: legacy 9-node tire thermal model (`src/models/physics.py`)
- Implemented and tested: 4-wheel vehicle thermal simulator (`src/models/vehicle_thermal.py`)
- Implemented and tested: high-fidelity simulation stack (`src/models/high_fidelity/`)
- Implemented and tested: no-data scenario/UQ harness and performance benchmarks (`scripts/`)
- Not implemented yet: training, evaluation, and sweep entrypoints (`src/train/*.py` are placeholders)

Canonical architecture notes are in [`simulation_model.md`](simulation_model.md).

## Requirements
- Python `>=3.14,<3.15`
- [`uv`](https://github.com/astral-sh/uv)

## Setup
```bash
UV_CACHE_DIR=/tmp/uv-cache-pitnn uv sync --dev
```

## Quick Validation
```bash
UV_CACHE_DIR=/tmp/uv-cache-pitnn uv run pitnn
UV_CACHE_DIR=/tmp/uv-cache-pitnn uv run pytest
UV_CACHE_DIR=/tmp/uv-cache-pitnn uv run ruff check .
UV_CACHE_DIR=/tmp/uv-cache-pitnn uv run mypy src
```

`pytest` is configured to skip `slow` tests by default. To include slow tests:

```bash
UV_CACHE_DIR=/tmp/uv-cache-pitnn uv run pytest -m "slow or not slow"
```

## High-Fidelity Harness
Run the synthetic no-data high-fidelity scenarios (deterministic + UQ):

```bash
UV_CACHE_DIR=/tmp/uv-cache-pitnn uv run python scripts/run_high_fidelity_no_data.py --preset dev --workers 1
```

Useful presets:
- `smoke`: quickest sanity run
- `dev`: development-time compromise
- `full`: highest default fidelity

Default outputs:
- JSON: `reports/results/high_fidelity_no_data_results.json`
- Markdown summary: `reports/results/high_fidelity_no_data_summary.md`

## Optional Native Acceleration
Build optional C++/pybind11 kernels:

```bash
UV_CACHE_DIR=/tmp/uv-cache-pitnn uv run python scripts/build_hf_diffusion_native.py
UV_CACHE_DIR=/tmp/uv-cache-pitnn uv run python scripts/build_hf_simulator_native.py
```

Enable native kernels at runtime:

```bash
export PITNN_USE_NATIVE_DIFFUSION=1
export PITNN_USE_NATIVE_SIMULATOR_KERNELS=1
```

## Benchmarks
```bash
UV_CACHE_DIR=/tmp/uv-cache-pitnn uv run python scripts/benchmark_high_fidelity.py
UV_CACHE_DIR=/tmp/uv-cache-pitnn uv run python scripts/benchmark_diffusion_kernel.py
UV_CACHE_DIR=/tmp/uv-cache-pitnn uv run python scripts/benchmark_simulator_kernels.py
```

`benchmark_high_fidelity.py` enforces deterministic equivalence and requires at least a 1% speedup in its parallel/native checks.

## Repository Layout
```text
PITNN/
  configs/
  data/
    raw/
    processed/
  native/
  reports/
    figures/
    results/
  scripts/
  src/
    data/
    models/
    pitnn/
    train/
    utils/
  tests/
```

## License
This project is licensed under the MIT License. See [`LICENSE`](LICENSE).
