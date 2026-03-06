# Plan: Add Sidewall + RR Split, 4-Wheel Dynamics, Slip Transients, Quasi-2D Patch, and Friction Heat Partition

## Summary
Implement the five requested realism upgrades in a phased, additive way that preserves current single-tire usage while introducing a new 4-wheel vehicle simulator.  
Rollout will be feature-flagged, validated against `/Users/nicolasschneider/MeineDokumente/Privat/Programmieren/F1-PITNN/reports/results/core_temp_simulation_calibrated.csv`, and accepted only with bounded output drift plus improved physics metrics.

## Locked Decisions
- API strategy: additive (keep `TireThermalSimulator` callable as today).
- Rollout style: phased flags with stage-by-stage validation.
- Contact patch depth: quasi-2D 3x3 grid.
- Validation source: existing calibrated CSV plus physics sanity checks.
- Acceptance mode: bounded drift + better physics.

## Public API / Interface Changes
1. Extend `/Users/nicolasschneider/MeineDokumente/Privat/Programmieren/F1-PITNN/src/models/physics.py` with additive interfaces:
- Add `t_sidewall_k` to tire thermal state (with compatibility defaults for legacy construction).
- Add `rolling_resistance_coeff` and RR split parameters.
- Add optional slip transient inputs: `slip_ratio_cmd`, `slip_angle_cmd_rad` (existing `slip_ratio` / `slip_angle_rad` remain valid).
- Add quasi-2D patch diagnostics fields:
`patch_pressure_grid_pa`, `patch_shear_grid_pa`, `patch_cell_area_m2`, `zone_friction_power_w`.
- Add friction partition parameters:
`friction_partition_adhesion`, `friction_partition_hysteresis`, `friction_partition_flash`.

2. Add new additive vehicle-level module:
- New file `/Users/nicolasschneider/MeineDokumente/Privat/Programmieren/F1-PITNN/src/models/vehicle_thermal.py`.
- New types:
`WheelId` (`FL`, `FR`, `RL`, `RR`), `VehicleInputs`, `VehicleParameters`, `VehicleState`, `VehicleDiagnostics`.
- New class:
`VehicleThermalSimulator` with `initial_state()`, `step()`, `simulate()` that internally runs 4 tire simulators and per-wheel load/slip allocation.

3. Export new types in:
`/Users/nicolasschneider/MeineDokumente/Privat/Programmieren/F1-PITNN/src/models/__init__.py`.

## Detailed Implementation Plan

## Phase 0: Baseline Snapshot and Validation Harness
1. Freeze baseline metrics from existing replay script output:
`/Users/nicolasschneider/MeineDokumente/Privat/Programmieren/F1-PITNN/reports/results/realism_stepwise_metrics.csv`.
2. Add a new staged runner:
`/Users/nicolasschneider/MeineDokumente/Privat/Programmieren/F1-PITNN/scripts/run_realism_v3_stepwise.py`.
3. Stage definitions:
`0=baseline`, `1=sidewall+RR`, `2=slip transients`, `3=quasi-2D patch`, `4=friction partition`, `5=vehicle 4-wheel coupling`.
4. For stages 1-4, replay the existing single-tire CSV profile.
5. For stage 5, add deterministic synthetic maneuvers for load-transfer checks (steady corner, brake-in-line, accel-in-line, combined brake+corner).

### Phase 0 Implementation Status (Completed on 2026-02-22)
- Added `/Users/nicolasschneider/MeineDokumente/Privat/Programmieren/F1-PITNN/scripts/run_realism_v3_stepwise.py` with stage definitions:
`0=baseline`, `1=sidewall_rr`, `2=slip_transients`, `3=quasi_2d_patch`, `4=friction_partition`, `5=vehicle_4wheel_coupling`.
- Implemented baseline snapshot freezing from:
`/Users/nicolasschneider/MeineDokumente/Privat/Programmieren/F1-PITNN/reports/results/realism_stepwise_metrics.csv`
into:
`/Users/nicolasschneider/MeineDokumente/Privat/Programmieren/F1-PITNN/reports/results/realism_v3_baseline_snapshot.csv`.
- Implemented deterministic single-tire replay harness for stages 0-4 against:
`/Users/nicolasschneider/MeineDokumente/Privat/Programmieren/F1-PITNN/reports/results/core_temp_simulation_calibrated.csv`.
- Implemented stage-5 deterministic synthetic 4-wheel maneuver validation for:
steady corner, brake-in-line, accel-in-line, and combined brake+corner.
- Generated Phase 0 outputs:
`/Users/nicolasschneider/MeineDokumente/Privat/Programmieren/F1-PITNN/reports/results/realism_v3_stepwise_metrics.csv`,
`/Users/nicolasschneider/MeineDokumente/Privat/Programmieren/F1-PITNN/reports/results/realism_v3_stepwise_metrics.txt`,
`/Users/nicolasschneider/MeineDokumente/Privat/Programmieren/F1-PITNN/reports/results/core_temp_simulation_realism_v3.csv`,
`/Users/nicolasschneider/MeineDokumente/Privat/Programmieren/F1-PITNN/reports/results/vehicle_4wheel_validation_summary.md`.
- Current Phase 0 harness behavior is intentionally additive: stages 2-4 run through the baseline single-tire path until Phase 2-4 physics blocks are implemented and activated.

## Phase 1: Sidewall Thermal Node + 85/15 Rolling Resistance Split
1. Add sidewall node thermodynamics in `/Users/nicolasschneider/MeineDokumente/Privat/Programmieren/F1-PITNN/src/models/physics.py`.
2. Add sidewall energy balance:
`m_sw*c_sw*dTsw = Q_rr_side + Q_hyst_side + G_carc_sw*(Tcarc-Tsw) + G_rim_sw*(Trim-Tsw) - h_sw_ext*A_sw*(Tsw-Tamb) - h_sw_int*A_sw*(Tsw-Tgas)`.
3. Compute rolling resistance power:
`Q_rr_total = Crr * Fz * Vx`.
4. Split RR power by default:
`Q_rr_belt_tread = 0.85*Q_rr_total`, `Q_rr_side = 0.15*Q_rr_total`.
5. Keep split configurable by parameters and enforce energy conservation in code path.
6. Route RR tread/belt portion into existing belt/tread heating path.

### Phase 1 Implementation Status (Completed on 2026-02-22)
- Extended tire state with sidewall temperature node `t_sidewall_k` in:
`/Users/nicolasschneider/MeineDokumente/Privat/Programmieren/F1-PITNN/src/models/physics.py`.
- Added backward-compatible vector decoding so legacy 8-thermal-node vectors remain readable (legacy vectors map sidewall temperature from carcass node).
- Added Phase 1 parameters and toggle:
`use_sidewall_rr_split_model`,
`rolling_resistance_coeff`,
`rr_belt_tread_fraction`,
`rr_sidewall_fraction`,
`sidewall_hysteresis_fraction`,
plus sidewall thermal mass/conductance/convection parameters.
- Implemented RR power and split with conservation guard:
`Q_rr_total = Crr * Fz * |Vx|`,
`Q_rr_belt_tread = f_belt * Q_rr_total`,
`Q_rr_sidewall = f_side * Q_rr_total`,
with runtime normalization of `(f_belt, f_side)` to enforce `f_belt + f_side = 1`.
- Implemented sidewall node energy balance coupling in ODE path:
carcass-to-sidewall, rim-to-sidewall, ambient convection, and gas-sidewall internal convection.
- Routed RR belt/tread power into belt/tread heating path and routed sidewall RR+hysteresis fraction into sidewall node.
- Added new diagnostics outputs for RR terms:
`rolling_resistance_total_w`,
`rolling_resistance_belt_tread_w`,
`rolling_resistance_sidewall_w`.
- Activated Phase 1 in staged runner:
`/Users/nicolasschneider/MeineDokumente/Privat/Programmieren/F1-PITNN/scripts/run_realism_v3_stepwise.py`
with `use_sidewall_rr_split_model = (stage >= 1)`.
- Added regression tests in:
`/Users/nicolasschneider/MeineDokumente/Privat/Programmieren/F1-PITNN/tests/test_physics_simulation.py`:
RR split conservation and sidewall thermal sink/lag behavior.
- Regenerated Phase 1 outputs in:
`/Users/nicolasschneider/MeineDokumente/Privat/Programmieren/F1-PITNN/reports/results/realism_v3_stepwise_metrics.csv`,
`/Users/nicolasschneider/MeineDokumente/Privat/Programmieren/F1-PITNN/reports/results/realism_v3_stepwise_metrics.txt`,
`/Users/nicolasschneider/MeineDokumente/Privat/Programmieren/F1-PITNN/reports/results/core_temp_simulation_realism_v3.csv`,
`/Users/nicolasschneider/MeineDokumente/Privat/Programmieren/F1-PITNN/reports/results/vehicle_4wheel_validation_summary.md`.

## Phase 2: Slip Transient Dynamics (Relaxation Length Model)
1. Add dynamic slip states per tire:
`kappa_dyn`, `alpha_dyn`.
2. Add first-order dynamics:
`dkappa/dt = (Vx/Lx)*(kappa_cmd - kappa_dyn)`,
`dalpha/dt = (Vx/Ly)*(alpha_cmd - alpha_dyn)`.
3. Use `kappa_dyn` and `alpha_dyn` in friction power and patch shear, not raw commands.
4. Add low-speed handling:
when `Vx < v_eps`, smoothly blend to algebraic slip to avoid stiffness/division artifacts.
5. Add parameters:
`relaxation_length_long_m`, `relaxation_length_lat_m`, `slip_transition_speed_mps`.

### Phase 2 Implementation Status (Completed on 2026-02-22)
- Added dynamic slip states to tire state in:
`/Users/nicolasschneider/MeineDokumente/Privat/Programmieren/F1-PITNN/src/models/physics.py`:
`kappa_dyn`, `alpha_dyn_rad`.
- Extended `TireInputs` with optional command channels:
`slip_ratio_cmd`, `slip_angle_cmd_rad`.
Legacy inputs remain valid and are used as fallback commands when optional fields are not provided.
- Added Phase 2 parameters:
`use_slip_transient_model`,
`relaxation_length_long_m`,
`relaxation_length_lat_m`,
`slip_transition_speed_mps`,
plus numerical safety cap `max_slip_relaxation_rate_per_s`.
- Implemented relaxation-length ODEs:
`dkappa/dt = (|Vx|/Lx) * (kappa_cmd - kappa_dyn)`,
`dalpha/dt = (|Vx|/Ly) * (alpha_cmd - alpha_dyn)`,
with capped gains for solver stability.
- Implemented low-speed blending to algebraic slip:
effective slips blend from dynamic state to command slip when speed approaches zero, controlled by `slip_transition_speed_mps`.
- Routed effective dynamic slips into friction/heating and wear drivers (instead of raw slip commands), including diagnostics slip-speed evaluation.
- Added diagnostics fields for effective slips:
`effective_slip_ratio`, `effective_slip_angle_rad`.
- Updated staged runner:
`/Users/nicolasschneider/MeineDokumente/Privat/Programmieren/F1-PITNN/scripts/run_realism_v3_stepwise.py`
to enable Phase 2 at `stage >= 2`.
- Added Phase 2 tests in:
`/Users/nicolasschneider/MeineDokumente/Privat/Programmieren/F1-PITNN/tests/test_physics_simulation.py`:
transient lag/tracking behavior and low-speed blending stability.
- Regenerated staged artifacts in:
`/Users/nicolasschneider/MeineDokumente/Privat/Programmieren/F1-PITNN/reports/results/realism_v3_stepwise_metrics.csv`,
`/Users/nicolasschneider/MeineDokumente/Privat/Programmieren/F1-PITNN/reports/results/realism_v3_stepwise_metrics.txt`,
`/Users/nicolasschneider/MeineDokumente/Privat/Programmieren/F1-PITNN/reports/results/core_temp_simulation_realism_v3.csv`,
`/Users/nicolasschneider/MeineDokumente/Privat/Programmieren/F1-PITNN/reports/results/vehicle_4wheel_validation_summary.md`.

## Phase 3: Quasi-2D Contact Patch (3x3 Zonal Model)
1. Replace scalar patch mechanics with 3x3 cell representation (width x length) while retaining backward-compatible aggregate outputs.
2. Construct per-cell pressure field using:
- total effective pressure from current gauge+carcass model,
- camber-induced width skew,
- longitudinal accel/brake-induced fore-aft skew,
- load normalization so sum(cell pressure*area) equals vertical load.
3. Construct per-cell shear from transient slips and local pressure.
4. Compute per-cell friction power and aggregate:
- per-width zones for thermal node coupling,
- total friction force/power for diagnostics.
5. Ensure constraints:
- all cell areas positive,
- all normal pressures non-negative,
- summed normal force matches target `Fz` within tolerance.

### Phase 3 Implementation Status (Completed on 2026-02-22)
- Added quasi-2D patch diagnostics fields in:
`/Users/nicolasschneider/MeineDokumente/Privat/Programmieren/F1-PITNN/src/models/physics.py`:
`patch_pressure_grid_pa`, `patch_shear_grid_pa`, `patch_cell_area_m2`, `zone_friction_power_w`.
- Added Phase 3 toggle/parameters:
`use_quasi_2d_patch_model`,
`patch_camber_pressure_skew`,
`patch_longitudinal_pressure_skew`,
`patch_pressure_min_fraction`,
`patch_pressure_norm_tol_n`.
- Implemented 3x3 contact patch construction with:
camber-based width skew, longitudinal-acceleration fore/aft skew, and force normalization so
`sum(cell_pressure * cell_area) ~= Fz`.
- Implemented per-cell shear stress and friction-power construction from effective (transient) slips and local pressure.
- Implemented per-width aggregation of friction power (`inner`, `middle`, `outer`) and wired it into thermal heating path when Phase 3 is enabled.
- Updated staged runner:
`/Users/nicolasschneider/MeineDokumente/Privat/Programmieren/F1-PITNN/scripts/run_realism_v3_stepwise.py`
to enable Phase 3 at `stage >= 3`.
- Added Phase 3 consistency test in:
`/Users/nicolasschneider/MeineDokumente/Privat/Programmieren/F1-PITNN/tests/test_physics_simulation.py`
covering cell-area positivity, pressure non-negativity, load-force consistency, and friction-power conservation.
- Regenerated staged artifacts in:
`/Users/nicolasschneider/MeineDokumente/Privat/Programmieren/F1-PITNN/reports/results/realism_v3_stepwise_metrics.csv`,
`/Users/nicolasschneider/MeineDokumente/Privat/Programmieren/F1-PITNN/reports/results/realism_v3_stepwise_metrics.txt`,
`/Users/nicolasschneider/MeineDokumente/Privat/Programmieren/F1-PITNN/reports/results/core_temp_simulation_realism_v3.csv`,
`/Users/nicolasschneider/MeineDokumente/Privat/Programmieren/F1-PITNN/reports/results/vehicle_4wheel_validation_summary.md`.

## Phase 4: Friction Heat Partition (Adhesion/Hysteresis/Flash)
1. Introduce partition of contact patch friction power:
`Q_fric_total = F_fric * v_slip`,
`Q_adh = c_adh*Q_fric_total`,
`Q_hys_surf = c_hys*Q_fric_total`,
`Q_flash = c_flash*Q_fric_total`,
with `c_adh + c_hys + c_flash = 1`.
2. Default literature-based coefficients:
`c_adh=0.45`, `c_hys=0.35`, `c_flash=0.20` (configurable).
3. Thermal routing:
- `Q_adh` and `Q_hys_surf` into tread surface/belt conduction path.
- `Q_flash` into fast surface sink/source term with short effective timescale (largely surface-local, limited core penetration).
4. Add a normalization guard that renormalizes coefficients to 1.0 if user config drifts.

### Phase 4 Implementation Status (Completed on 2026-02-22)
- Added Phase 4 toggle/parameters in:
`/Users/nicolasschneider/MeineDokumente/Privat/Programmieren/F1-PITNN/src/models/physics.py`:
`use_friction_partition_model`,
`friction_partition_adhesion`,
`friction_partition_hysteresis`,
`friction_partition_flash`,
`flash_surface_time_constant_s`.
- Implemented coefficient normalization guard:
negative coefficients are clipped to zero and coefficients are renormalized to sum to 1.0 (`_friction_partition_coefficients`).
- Implemented partitioned friction terms:
`Q_adh`, `Q_hys_surf`, `Q_flash` from per-zone friction power.
- Implemented thermal routing behavior:
`Q_adh` + `Q_hys_surf` are routed to standard surface/belt path;
`Q_flash` is routed through a fast surface-local sink-limited term (via `flash_surface_time_constant_s`) to reduce deep penetration.
- Added friction partition diagnostics in:
`/Users/nicolasschneider/MeineDokumente/Privat/Programmieren/F1-PITNN/src/models/physics.py`:
`friction_power_total_w`,
`friction_power_adhesion_w`,
`friction_power_hysteresis_w`,
`friction_power_flash_w`.
- Updated staged runner:
`/Users/nicolasschneider/MeineDokumente/Privat/Programmieren/F1-PITNN/scripts/run_realism_v3_stepwise.py`
to enable Phase 4 at `stage >= 4`.
- Added Phase 4 tests in:
`/Users/nicolasschneider/MeineDokumente/Privat/Programmieren/F1-PITNN/tests/test_physics_simulation.py`:
partition normalization/conservation and flash-routing sanity over a transient run.
- Regenerated staged artifacts in:
`/Users/nicolasschneider/MeineDokumente/Privat/Programmieren/F1-PITNN/reports/results/realism_v3_stepwise_metrics.csv`,
`/Users/nicolasschneider/MeineDokumente/Privat/Programmieren/F1-PITNN/reports/results/realism_v3_stepwise_metrics.txt`,
`/Users/nicolasschneider/MeineDokumente/Privat/Programmieren/F1-PITNN/reports/results/core_temp_simulation_realism_v3.csv`,
`/Users/nicolasschneider/MeineDokumente/Privat/Programmieren/F1-PITNN/reports/results/vehicle_4wheel_validation_summary.md`.

## Phase 5: True 4-Wheel Coupling via Vehicle Dynamics
1. Add `/Users/nicolasschneider/MeineDokumente/Privat/Programmieren/F1-PITNN/src/models/vehicle_thermal.py`.
2. Implement per-wheel vertical loads using:
- static weight distribution,
- longitudinal load transfer (`ax`, CG height, wheelbase),
- lateral load transfer (`ay`, CG height, track widths),
- aero downforce split front/rear with speed dependence.
3. Implement brake power distribution by brake bias:
- front axle fraction and equal split per wheel on axle by default.
4. Implement per-wheel slip commands:
- steering/yaw and axle kinematics to derive wheel-level `alpha_cmd`,
- wheel rotational state or commanded torque to derive `kappa_cmd`.
5. Feed each wheel into one tire simulator instance (same model, per-wheel parameters permitted).
6. Return `VehicleDiagnostics` with per-wheel temperatures, pressures, loads, slips, and power terms.

### Phase 5 Implementation Status (Completed on 2026-02-22)
- Added new module:
`/Users/nicolasschneider/MeineDokumente/Privat/Programmieren/F1-PITNN/src/models/vehicle_thermal.py`.
- Added additive vehicle-level types:
`WheelId` (`FL`, `FR`, `RL`, `RR`),
`VehicleInputs`,
`VehicleParameters`,
`VehicleState`,
`VehicleDiagnostics`.
- Added new class:
`VehicleThermalSimulator` with additive API:
`initial_state()`, `step()`, `simulate()`, and `diagnostics()`.
- Implemented per-wheel load model with:
static front/rear weight split,
longitudinal transfer (`ax`, CG height, wheelbase),
lateral transfer (`ay`, CG height, track widths),
aero downforce split front/rear.
- Implemented brake and drive power distribution:
front/rear bias and equal wheel split on each axle.
- Implemented per-wheel slip command generation:
wheel-level `alpha_cmd` from steering/yaw-track kinematics and
wheel-level `kappa_cmd` from power demand and optional wheel-speed blending.
- Coupled all four wheels into independent tire simulators (`TireThermalSimulator`) with per-wheel `TireInputs`.
- Exported new vehicle types in:
`/Users/nicolasschneider/MeineDokumente/Privat/Programmieren/F1-PITNN/src/models/__init__.py`.
- Replaced stage-5 placeholder validation path in:
`/Users/nicolasschneider/MeineDokumente/Privat/Programmieren/F1-PITNN/scripts/run_realism_v3_stepwise.py`
to run deterministic maneuvers through `VehicleThermalSimulator` and emit real 4-wheel diagnostics.
- Added dedicated tests in:
`/Users/nicolasschneider/MeineDokumente/Privat/Programmieren/F1-PITNN/tests/test_vehicle_thermal.py`:
load conservation, transfer sign checks, brake-bias consistency, and long-run finite/bounded state checks.
- Regenerated staged artifacts in:
`/Users/nicolasschneider/MeineDokumente/Privat/Programmieren/F1-PITNN/reports/results/realism_v3_stepwise_metrics.csv`,
`/Users/nicolasschneider/MeineDokumente/Privat/Programmieren/F1-PITNN/reports/results/realism_v3_stepwise_metrics.txt`,
`/Users/nicolasschneider/MeineDokumente/Privat/Programmieren/F1-PITNN/reports/results/core_temp_simulation_realism_v3.csv`,
`/Users/nicolasschneider/MeineDokumente/Privat/Programmieren/F1-PITNN/reports/results/vehicle_4wheel_validation_summary.md`.

## Phase 6: Backward Compatibility and Migration Rules
1. Preserve existing single-tire public calls:
`TireThermalSimulator.initial_state`, `step`, `simulate`, `diagnostics`.
2. Keep legacy callers valid when only old `TireInputs` fields are provided.
3. Add compatibility constructors for state vectors if new dimensions are introduced.
4. Keep existing report scripts functional; add new script rather than replacing old paths.

### Phase 6 Implementation Status (Completed on 2026-02-22)
- Preserved existing single-tire public API (`initial_state`, `step`, `simulate`, `diagnostics`) in:
`/Users/nicolasschneider/MeineDokumente/Privat/Programmieren/F1-PITNN/src/models/physics.py`.
- Kept legacy caller behavior for `TireInputs`:
new fields remain optional and existing call sites with old fields continue to run unchanged.
- Added explicit compatibility constructors in:
`/Users/nicolasschneider/MeineDokumente/Privat/Programmieren/F1-PITNN/src/models/physics.py`:
`TireState.from_legacy_vector_9(...)` and `TireState.from_legacy_vector_10(...)`,
plus backward-compatible decoding in `TireState.from_vector(...)`.
- Added compatibility tests in:
`/Users/nicolasschneider/MeineDokumente/Privat/Programmieren/F1-PITNN/tests/test_physics_simulation.py`
to verify legacy vector restoration behavior with new state dimensions.
- Verified existing report scripts remain functional:
`/Users/nicolasschneider/MeineDokumente/Privat/Programmieren/F1-PITNN/scripts/run_realism_stepwise.py`
and
`/Users/nicolasschneider/MeineDokumente/Privat/Programmieren/F1-PITNN/scripts/run_realism_v3_stepwise.py`
both execute successfully.

## Phase 7: Tests and Validation Gates
1. Extend `/Users/nicolasschneider/MeineDokumente/Privat/Programmieren/F1-PITNN/tests/test_physics_simulation.py` with:
- sidewall node finite behavior and thermal lag checks,
- RR split conservation (`Q_rr_side + Q_rr_belt_tread == Q_rr_total`),
- slip transient lag response and stability at low speed,
- quasi-2D patch force consistency,
- friction partition conservation and routing sanity.
2. Add `/Users/nicolasschneider/MeineDokumente/Privat/Programmieren/F1-PITNN/tests/test_vehicle_thermal.py` with:
- load sum consistency (`sum(Fz_wheels) ~= total vertical force`),
- correct sign of front-rear transfer under braking/accel,
- correct sign of left-right transfer under lateral accel,
- brake bias distribution consistency,
- all wheel states finite and bounded in long runs.
3. Add staged metric checks in `run_realism_v3_stepwise.py`.
4. Acceptance thresholds:
- no NaN/Inf anywhere,
- single-tire replay final core drift within `±4.0 C` vs baseline,
- single-tire replay final pressure drift within `±0.12 bar(g)` vs baseline,
- `patch_pressure_corr <= -0.40` by phase 4+,
- internal HTC hard-cap clipping fraction `< 0.05` by phase 5,
- 4-wheel load conservation error `< 0.5%`.

### Phase 7 Implementation Status (Completed on 2026-02-22)
- Extended physics simulation tests in:
`/Users/nicolasschneider/MeineDokumente/Privat/Programmieren/F1-PITNN/tests/test_physics_simulation.py`
to cover sidewall, RR split conservation, slip transients, quasi-2D patch consistency, friction partition conservation/routing, and legacy state compatibility.
- Added vehicle-level tests in:
`/Users/nicolasschneider/MeineDokumente/Privat/Programmieren/F1-PITNN/tests/test_vehicle_thermal.py`
for load conservation, transfer signs, brake-bias consistency, and long-run boundedness.
- Added staged validation-gate evaluation in:
`/Users/nicolasschneider/MeineDokumente/Privat/Programmieren/F1-PITNN/scripts/run_realism_v3_stepwise.py`
(`evaluate_validation_gates(...)`), with gate outputs written to:
`/Users/nicolasschneider/MeineDokumente/Privat/Programmieren/F1-PITNN/reports/results/realism_v3_validation_gates.csv`
and appended to:
`/Users/nicolasschneider/MeineDokumente/Privat/Programmieren/F1-PITNN/reports/results/realism_v3_stepwise_metrics.txt`.
- Current gate status from generated artifacts:
all Phase 7 acceptance checks are passing (`6/6`).

## Phase 8: Documentation and Deliverables
1. Update `/Users/nicolasschneider/MeineDokumente/Privat/Programmieren/F1-PITNN/simulation_model.md` with new governing equations and coupling blocks.
2. Add implementation notes and parameter tables in:
`/Users/nicolasschneider/MeineDokumente/Privat/Programmieren/F1-PITNN/reports/results/physics_realism_upgrade_report.md`.
3. Generate new outputs:
- `/Users/nicolasschneider/MeineDokumente/Privat/Programmieren/F1-PITNN/reports/results/realism_v3_stepwise_metrics.csv`
- `/Users/nicolasschneider/MeineDokumente/Privat/Programmieren/F1-PITNN/reports/results/realism_v3_stepwise_metrics.txt`
- `/Users/nicolasschneider/MeineDokumente/Privat/Programmieren/F1-PITNN/reports/results/core_temp_simulation_realism_v3.csv`
- `/Users/nicolasschneider/MeineDokumente/Privat/Programmieren/F1-PITNN/reports/results/vehicle_4wheel_validation_summary.md`

## Assumptions and Defaults
- No new experimental dataset is required for implementation; coefficients use literature-prior defaults and are configurable.
- Existing calibrated CSV remains the primary acceptance anchor for single-tire behavior.
- 4-wheel validation uses deterministic synthetic maneuvers until real per-wheel telemetry is wired.
- Feature flags remain available to isolate each physics block during tuning/debugging.
- Runtime target stays compatible with current Python stack and existing tests.
