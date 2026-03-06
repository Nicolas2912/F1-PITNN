# Simulation Model

This is the canonical architecture note for the simulation code in this repository.

## Current Model Stack

The repository contains two additive simulation paths:

1. Legacy compatibility path in `src/models/physics.py`
   - pressure-coupled 9-node thermal network
   - nodes: surface inner/middle/outer, belt, carcass, gas, rim, brake, sidewall
   - kept stable for existing `TireThermalSimulator` and `VehicleThermalSimulator` callers
2. High-fidelity path in `src/models/high_fidelity/`
   - 2D radial x circumferential thermal field
   - moving contact-patch source window
   - viscoelastic hysteresis model with Prony branches and WLF shift
   - road and rim boundary models with explicit heat partition
   - wheel-level force/slip iteration and 4-wheel vehicle orchestration
   - seeded UQ with LHS envelopes and Sobol ranking

## High-Fidelity Architecture

### Thermal State

The high-fidelity tire uses a 2D field `T[r, theta]` plus reduced 9-node diagnostics for compatibility:

- surface inner/middle/outer
- belt
- carcass/core
- gas
- rim
- brake
- sidewall proxy

The reduced nodes are derived from the field after each step so legacy-style summaries remain available.

The reported `core` node is not treated as a full-thickness carcass average. In the absence of proprietary instrument placement data, it is modeled as an embedded under-tread carcass / belt-package probe with first-order sensor lag. That matches how race-tire "core" telemetry is commonly interpreted and produces more realistic separation between surface and bulk response on stint-length horizons.

### Materials and Hysteresis

`src/models/high_fidelity/materials.py` models hysteresis from:

- excitation frequency from wheel rotation plus slip speed
- WLF time-temperature shift
- generalized Maxwell / Prony loss modulus
- volumetric dissipation `q_hyst >= 0`

This model is intended for physically consistent trend prediction, not compound-specific truth without calibration data.

### Thermal Solver

`src/models/high_fidelity/thermal_solver.py` uses:

- radial cells: default `24`
- circumferential cells: default `72`
- internal time step: default `0.01 s`
- explicit first-order upwind advection in `theta`
- implicit diffusion solve in `r, theta`

Acceptance is based on worst-step energy closure, not average residual.

### Boundaries and Partition

`src/models/high_fidelity/boundary_conditions.py` models:

- tire-road friction partition with `eta_tire`
- tire-road contact conductance `h_cp`
- tire-rim bead conductance `h_c_bead`
- 2-node road slab: surface and subsurface
- rim thermal mass and ambient cooling
- explicit brake-power leakage fractions into tire inner liner and rim

### Wheel and Vehicle Coupling

`src/models/high_fidelity/wheel_coupling.py` solves for effective slip ratio and slip angle under:

- wheel torque balance
- lateral-force target matching
- bounded slip envelopes for deterministic fallback

`src/models/high_fidelity/vehicle_simulator.py` applies:

- aero downforce split
- longitudinal and lateral load transfer
- steering and yaw kinematics
- per-wheel drive and brake torque allocation

### UQ Outputs

`src/models/high_fidelity/uq.py` and `scripts/run_high_fidelity_no_data.py` publish:

- deterministic scenario traces
- uncertainty envelopes
- Sobol sensitivity ranking

Published high-fidelity outputs should include surface-sensitive metrics, because core temperature alone can be too slow-moving over short synthetic maneuvers.

## Scenario Coverage

The reporting harness includes:

- short benchmark maneuvers for convergence, energy closure, and sign checks
- a longer deterministic `long_stint` scenario intended to expose whether the current core proxy actually moves on a longer horizon

The long-stint scenario is kept out of the default UQ sweep because it is a deterministic validation case, not a practical uncertainty benchmark at the current runtime cost.

## Acceptance Criteria Mapping

The intended project-level checks are:

1. Physics consistency: worst per-step energy closure residual below 1 percent in benchmark scenarios.
2. Numerical robustness: no NaN or Inf in deterministic traces, diagnostics, UQ envelopes, or Sobol outputs.
3. Coupling realism: wheel-step convergence rate above 99 percent across benchmark scenarios.
4. UQ completeness: uncertainty bands and Sobol ranking published together.
5. Compatibility: legacy public APIs remain callable without changes.
6. Documentation: this file and the README match the actual implementation.

## Assumptions

- This is an offline model, not a real-time estimator.
- Without proprietary tire data, outputs are bounded physical hypotheses rather than validated race-engineering truth.
- Synthetic scenarios are used to validate signs, conservation, stability, and parameter sensitivity.
- Surface, belt, and core temperatures can evolve on very different timescales; short scenarios should not be judged on core response alone.
- The `core` proxy is intentionally biased toward the heated outer carcass / belt region and away from cavity-gas temperature, because that is the closest defensible assumption for typical motorsport probe data when exact sensor packaging is unavailable.
