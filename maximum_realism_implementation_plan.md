# Maximum-Realism Tire Thermal Model (No Data) — Prioritized Implementation Plan

## Summary
This plan targets **offline maximum physical realism** with **additive backward compatibility** to your current APIs.
The end-state is a coupled model with: 2D thermal field (radial × circumferential), moving contact-patch heat sources, viscoelastic frequency/temperature loss modeling (Prony + WLF), road/rim contact resistance, closed-loop wheel force/slip coupling, and mandatory uncertainty quantification (UQ) so outputs are delivered as confidence bands instead of unsupported single-point truths.

Target Markdown artifact created: `/Users/nicolasschneider/MeineDokumente/Privat/Programmieren/F1-PITNN/maximum_realism_implementation_plan.md`

## Prioritized List (Execution Order)

1. **P0 — Correctness and consistency baseline (must-do before fidelity upgrades).**
Implement friction-energy consistency so quasi-2D and non-quasi paths use one unified power accounting rule in `/Users/nicolasschneider/MeineDokumente/Privat/Programmieren/F1-PITNN/src/models/physics.py`.
Enable rolling-resistance heating by default through a global RR path (not gated only by sidewall split).
Update documentation mismatch (`8-node` vs current state model) in `/Users/nicolasschneider/MeineDokumente/Privat/Programmieren/F1-PITNN/README.md`.
Acceptance: existing tests pass, new invariants prove energy consistency across feature toggles.

2. **P1 — Introduce high-fidelity module skeleton (additive, non-breaking).**
Add new package `/Users/nicolasschneider/MeineDokumente/Privat/Programmieren/F1-PITNN/src/models/high_fidelity/` with interfaces, types, and adapters while keeping `/Users/nicolasschneider/MeineDokumente/Privat/Programmieren/F1-PITNN/src/models/physics.py` intact.
Acceptance: old API unchanged; new simulator instantiates and runs a no-op thermal step.

3. **P2 — Material physics upgrade (Prony + WLF + temperature-dependent transport).**
Implement generalized Maxwell/Prony representation for loss modulus and WLF time-temperature shift in `/Users/nicolasschneider/MeineDokumente/Privat/Programmieren/F1-PITNN/src/models/high_fidelity/materials.py`.
Use this to compute hysteresis volumetric source per cell and frequency band proxy from wheel kinematics.
Acceptance: unit tests validate monotonic WLF shift behavior and non-negative dissipated power across operating ranges.

4. **P3 — 2D thermal solver (radial × circumferential) with moving source window.**
Implement finite-volume grid solver in `/Users/nicolasschneider/MeineDokumente/Privat/Programmieren/F1-PITNN/src/models/high_fidelity/thermal_solver.py`.
Method choice (fixed): operator splitting with first-order upwind advection in θ and implicit backward-Euler diffusion solve; sparse linear solve via SciPy.
Default grid (fixed): `N_r=24`, `N_theta=72`, nonuniform radial spacing biased near tread and belt.
Acceptance: energy residual per step < 1% and stable long runs at `dt_internal=0.01 s`.

5. **P4 — Contact and boundary realism (road/rim resistances + partition).**
Add explicit tire-road heat partition `eta_tire` and tire-road contact conductance `h_cp` in `/Users/nicolasschneider/MeineDokumente/Privat/Programmieren/F1-PITNN/src/models/high_fidelity/boundary_conditions.py`.
Add rim/bead thermal contact conductance `h_c_bead` (pressure-sensitive option) in same module.
Road model (fixed): 2-node road slab (`surface`, `subsurface`) with conductive coupling.
Acceptance: sensitivity tests show expected trends for `eta_tire`, `h_cp`, `h_c_bead`, and road temperature.

6. **P5 — Closed-loop tire force/slip/thermal coupling at wheel level.**
Replace heuristic slip-command-only coupling with force-equilibrium iteration in `/Users/nicolasschneider/MeineDokumente/Privat/Programmieren/F1-PITNN/src/models/high_fidelity/wheel_coupling.py`.
Use iterative solve per wheel step: guess slip → compute forces/heat → update slip from torque balance until convergence.
Acceptance: convergence within max 8 iterations for standard scenarios and deterministic fallback behavior when non-convergent.

7. **P6 — Vehicle-level coupling upgrade (4-wheel high-fidelity orchestrator).**
Add `/Users/nicolasschneider/MeineDokumente/Privat/Programmieren/F1-PITNN/src/models/high_fidelity/vehicle_simulator.py` that couples 4 wheels, aero-load split, longitudinal/lateral transfer, steering/yaw kinematics, and per-wheel thermal fields.
Keep current `/Users/nicolasschneider/MeineDokumente/Privat/Programmieren/F1-PITNN/src/models/vehicle_thermal.py` as compatibility layer/adaptor.
Acceptance: load conservation, sign checks, and thermal asymmetry behavior pass deterministic maneuver suite.

8. **P7 — UQ-first outputs (mandatory for no-data realism claim).**
Add `/Users/nicolasschneider/MeineDokumente/Privat/Programmieren/F1-PITNN/src/models/high_fidelity/uq.py` for LHS sampling + Sobol indices.
Fixed workflow: LHS screening `N=400`, Sobol main run `N=2048`, output median and 5/50/95 percentile envelopes.
Acceptance: reproducible seeded runs and automatic ranked sensitivity report for dominant uncertainty drivers.

9. **P8 — Scenario harness and reporting deliverables.**
Add `/Users/nicolasschneider/MeineDokumente/Privat/Programmieren/F1-PITNN/scripts/run_high_fidelity_no_data.py` and `/Users/nicolasschneider/MeineDokumente/Privat/Programmieren/F1-PITNN/scripts/report_high_fidelity_no_data.py`.
Outputs: scenario traces, uncertainty bands, energy-balance diagnostics, and physical plausibility checks under synthetic maneuvers.
Acceptance: one-command generation of `reports/results/high_fidelity_no_data_summary.md` with full metrics.

## Public APIs, Interfaces, and Types (Additions)

1. Add `HighFidelityTireInputs` in `/Users/nicolasschneider/MeineDokumente/Privat/Programmieren/F1-PITNN/src/models/high_fidelity/types.py` with explicit environment channels: `wind_mps`, `humidity_rel`, `solar_w_m2`, `road_surface_temp_k`, `road_bulk_temp_k`.
2. Add `HighFidelityTireState` with field tensors for `T[r,theta,layer]`, plus reduced diagnostics (`core_k`, `surface_mid_k`, `gas_k`, `rim_k`).
3. Add `HighFidelityMaterialParameters` with Prony branches, WLF constants, and transport priors.
4. Add `HighFidelityBoundaryParameters` with `eta_tire`, `h_cp`, `h_c_bead`, `h_out_model`.
5. Add `HighFidelityTireSimulator` methods: `initial_state()`, `step()`, `simulate()`, `diagnostics()`.
6. Add `HighFidelityVehicleSimulator` methods: `initial_state()`, `step()`, `simulate()`, `diagnostics()`.
7. Keep existing exports from `/Users/nicolasschneider/MeineDokumente/Privat/Programmieren/F1-PITNN/src/models/__init__.py`; add new exports without removing old names.

## Test Cases and Scenarios

1. Add material tests in `/Users/nicolasschneider/MeineDokumente/Privat/Programmieren/F1-PITNN/tests/test_hf_materials.py` for WLF shift monotonicity, Prony positivity, and dissipation non-negativity.
2. Add thermal solver tests in `/Users/nicolasschneider/MeineDokumente/Privat/Programmieren/F1-PITNN/tests/test_hf_thermal_solver.py` for stability, CFL-safe advection behavior, and diffusion conservation on synthetic fields.
3. Add boundary/contact tests in `/Users/nicolasschneider/MeineDokumente/Privat/Programmieren/F1-PITNN/tests/test_hf_boundaries.py` for `eta_tire` split conservation and sign-correct road/rim fluxes.
4. Add coupling tests in `/Users/nicolasschneider/MeineDokumente/Privat/Programmieren/F1-PITNN/tests/test_hf_wheel_coupling.py` for iteration convergence and physically bounded slips.
5. Add vehicle tests in `/Users/nicolasschneider/MeineDokumente/Privat/Programmieren/F1-PITNN/tests/test_hf_vehicle.py` for 4-wheel load consistency, turn asymmetry, braking front-load transfer, and finite-state long runs.
6. Add UQ tests in `/Users/nicolasschneider/MeineDokumente/Privat/Programmieren/F1-PITNN/tests/test_hf_uq.py` for deterministic seeded sampling and stable quantile/sensitivity outputs.
7. Add end-to-end regression harness in `/Users/nicolasschneider/MeineDokumente/Privat/Programmieren/F1-PITNN/tests/test_hf_end_to_end.py` for synthetic stint profiles and report artifact schema checks.

## Acceptance Criteria (Project-Level)

1. Physics consistency: per-step energy closure residual < 1% in benchmark scenarios.
2. Numerical robustness: no NaN/Inf across all deterministic and UQ runs.
3. Coupling realism: closed-loop slip/force iteration converges in > 99% scenario steps.
4. UQ completeness: final outputs published as uncertainty bands and Sobol ranking, not point estimate only.
5. Compatibility: existing `TireThermalSimulator` and `VehicleThermalSimulator` API callers keep working unchanged.
6. Documentation: new architecture and assumptions documented in one canonical file and README aligned with actual model.

## Assumptions and Defaults (Locked)

1. Runtime target is offline high-fidelity, not real-time.
2. API strategy is additive compatibility, not breaking redesign.
3. No proprietary or experimental tire data is required; all uncertain parameters are modeled as priors with UQ.
4. Default solver internals: `dt_internal=0.01 s`, `N_r=24`, `N_theta=72`, operator-split advection/diffusion.
5. Default UQ sizes: LHS 400, Sobol 2048, fixed reproducible seed.
6. Default scenario pack includes steady corner, straight-line braking, straight-line acceleration, combined brake+corner, cooldown.
7. Claims are limited to physically consistent trends and uncertainty-bounded outputs, not absolute truth values.

## Implementation Sequence (No Decision Left for Implementer)

1. Complete P0 entirely and merge only after invariants pass.
2. Build P1 interfaces and adapters before any heavy solver work.
3. Implement P2 and P3 together, then validate energy closure.
4. Add P4 boundary/contact terms and revalidate conservation.
5. Integrate P5 wheel iterative coupling; then P6 full vehicle orchestrator.
6. Implement P7 UQ framework and wire it into simulation pipeline.
7. Finish P8 reporting scripts and final documentation alignment.
8. Freeze defaults and publish baseline no-data benchmark artifacts for future comparisons.
