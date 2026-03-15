# PITNN

Physics-informed tire thermal simulation project centered on the high-fidelity 4-wheel thermal simulator.

## Current Status
- Implemented and tested: 4-wheel vehicle thermal simulator (`src/models/vehicle_thermal.py`)
- Implemented and tested: high-fidelity simulation stack (`src/models/high_fidelity/`)
- Implemented and tested: no-data scenario/UQ harness (`scripts/run_high_fidelity_no_data.py`)

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
.venv/bin/python scripts/run_high_fidelity_no_data.py \
  --output-stem reports/results/run_2026-03-14_full_surrogate
```

Default outputs:
- JSON: `reports/results/high_fidelity_no_data_results.json`
- Markdown summary: `reports/results/high_fidelity_no_data_summary.md`
- `--output-stem foo/bar/run_name` writes `foo/bar/run_name.json` and `foo/bar/run_name.md`

The harness now defaults to the native accelerated path when the compiled extensions are available:
- native diffusion enabled
- ADI solver mode enabled
- native simulator kernels enabled
- Extra Trees surrogate enabled for Sobol UQ
- worker count defaults to `min(8, cpu_count)`

## Optional Native Acceleration
Build optional C++/pybind11 kernels:

```bash
UV_CACHE_DIR=/tmp/uv-cache-pitnn uv run python scripts/build_hf_diffusion_native.py
UV_CACHE_DIR=/tmp/uv-cache-pitnn uv run python scripts/build_hf_simulator_native.py
```

## Repository Layout
```text
PITNN/
  configs/
  native/
  reports/results/
  scripts/
  src/
    models/
    pitnn/
  tests/
```

## License
This project is licensed under the MIT License. See [`LICENSE`](LICENSE).
