# PITNN

Physics-informed neural network project scaffold for 5-second-ahead tire core temperature prediction.

## Environment
- Python: `3.14.x` (newest currently compatible baseline used here)
- Package manager: [`uv`](https://github.com/astral-sh/uv)

## Setup
```bash
UV_CACHE_DIR=/tmp/uv-cache-pitnn uv sync --dev
```

## Quick Checks
```bash
UV_CACHE_DIR=/tmp/uv-cache-pitnn uv run python -c "import fastf1, hydra, numpy, pandas, pydantic, requests_cache, retry_requests, rich, tqdm"
UV_CACHE_DIR=/tmp/uv-cache-pitnn uv run pytest
UV_CACHE_DIR=/tmp/uv-cache-pitnn uv run ruff check .
UV_CACHE_DIR=/tmp/uv-cache-pitnn uv run mypy src
```

## Physics Tire Simulation (9-Node)
`src/models/physics.py` implements a pressure-coupled 9-node thermal model (surface I/M/O, belt, carcass,
gas, rim, brake, sidewall) with:
- wear-dependent tread thickness and mass
- pressure/volume feedback for contact patch updates
- `P*dV` compression work in the gas node
- external and internal convection correlations

Minimal usage:
```python
from models.physics import TireInputs, TireThermalSimulator, celsius_to_kelvin

sim = TireThermalSimulator()
state = sim.initial_state(ambient_temp_k=celsius_to_kelvin(30.0))
u = TireInputs(
    speed_mps=70.0,
    wheel_angular_speed_radps=210.0,
    normal_load_n=3800.0,
    slip_ratio=0.08,
    slip_angle_rad=0.05,
    brake_power_w=3000.0,
    ambient_temp_k=celsius_to_kelvin(30.0),
    track_temp_k=celsius_to_kelvin(44.0),
)
for _ in range(300):
    state = sim.step(state, u, dt_s=0.1)

print("Core temperature [C]:", state.core_temperature_c)
print("Pressure [bar(g)]:", sim.diagnostics(state, u).dynamic_pressure_bar_gauge)
```

## High-Fidelity Simulation
`src/models/high_fidelity/` adds the non-breaking high-fidelity path:
- 2D radial x circumferential thermal solver
- Prony + WLF hysteresis model
- road and rim boundary heat transfer
- brake heat leakage into tire/rim
- under-tread core-probe assumption with finite sensor lag
- closed-loop wheel slip/force coupling
- 4-wheel vehicle orchestration
- uncertainty envelopes and Sobol ranking via `scripts/run_high_fidelity_no_data.py`
- deterministic long-stint validation alongside the short benchmark maneuvers

The canonical architecture and modeling assumptions live in
[`simulation_model.md`](simulation_model.md).

## Repository Layout
```text
PITNN/
  data/
    raw/
    processed/
  notebooks/
  src/
    data/
    models/
    train/
    utils/
  configs/
  reports/
    figures/
    results/
  tests/
```
