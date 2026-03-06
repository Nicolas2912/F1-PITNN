# Physics-Informed Neural Network for Tire Temperature Prediction

## 1. Project Summary
Build a physics-informed ML system that predicts **relative tire core temperature rise over 5 seconds** (rolling horizon) from motorsport telemetry and weather context.

This project combines:
- **Data-driven sequence modeling** (LSTM and/or Transformer)
- **Physics constraints** from tire thermodynamics to prevent physically invalid predictions

Primary objective:
- Predict `ΔT_core(t→t+5s) = T_core(t + 5s) - T_core(t)` accurately enough for actionable race-engineering support while enforcing basic thermodynamic consistency.

## 2. Problem Statement
Pure black-box models can produce impossible predictions (e.g., tire cooling during sustained burnout-like energy input). In motorsport, these failures are unacceptable because model outputs directly affect strategy decisions.

This project constrains learning with physics so predictions remain both:
- **Accurate** (statistical fit to data)
- **Physically plausible** (thermodynamic sanity)

## 3. Scope and Non-Goals
### In Scope
- Collect telemetry from FastF1 sessions
- Enrich telemetry with weather from Open-Meteo
- Train sequence models for 5-second-ahead tire core temperature rise prediction
- Add physics-based loss terms and hard/soft constraints
- Compare unconstrained vs physics-informed performance
- Provide reproducible training/evaluation pipeline

### Out of Scope
- Full vehicle dynamics simulation
- CFD/aero estimation from first principles
- Real-time in-car deployment for production race operations
- Absolute tire model calibration across all compounds and series

## 4. Critical Assumptions and Limitations
### Explicit Assumption (Required)
The model assumes tire core temperature is a function of available telemetry and weather features.

### Key Limitation (Must Be Stated Clearly)
This is incomplete in reality. **Aerodynamic setup, suspension kinematics, tire construction details, setup choices (camber/toe), tire pressure state, brake bias, and track surface evolution** strongly influence thermal behavior and are not fully observed in the public data.

Critical refinement:
- **Tire pressure** is one of the dominant thermal-state variables (thermo-mechanical coupling and hysteresis behavior). Without pressure telemetry, pressure effects are treated as latent noise or quasi-constant offsets.
- **Brake bias** redistributes thermal load front vs rear under braking. Missing bias data limits axle-specific thermal accuracy.

Proxy strategy:
- FastF1 does not expose live tire pressure. Where possible, use stint/lap phase and pit-stop tire change events as coarse proxies for pressure/degradation state.
- Use braking-energy distribution proxies (front/rear thermal trends if available, braking intensity context) as weak surrogates for brake-bias effects.

Consequences:
- Prediction ceiling is bounded by unobserved variables
- Domain shift risk is high across events/cars/drivers
- Model should be framed as a decision-support estimate, not a physics-complete truth

## 5. Data Specification
## 5.1 Sources
- **FastF1 telemetry** (public):
  - Speed
  - Throttle
  - Brake
  - RPM / gear (if available)
  - Position / distance / time
  - Lateral and longitudinal acceleration proxies (derived if needed)
- **Open-Meteo API**:
  - Ambient temperature
  - Track-relevant weather proxies (wind speed, humidity, cloud cover, solar radiation if available)

## 5.2 Target Variable
- Primary target: `ΔT_core(t→t+5s) = T_core(t + 5s) - T_core(t)`
- Secondary/derived target (optional): `T_core(t + 5s) = T_core(t) + ΔT_core(t→t+5s)`

If direct core temp label is unavailable in selected dataset split, define an interim proxy target and explicitly document reduced validity.
If absolute temperature labels are weak/noisy, prioritize robust delta-label quality over absolute-value fidelity.

## 5.3 Sampling and Alignment
- Standardize to fixed timestep (e.g., 10 Hz)
- Synchronize telemetry and weather by timestamp
- Handle missing data with forward-fill + mask features
- Segment by lap/session/stint

## 5.4 Feature Set
### Raw Features
- Speed
- Throttle [%]
- Brake [% or binary]
- Steering (if available)
- Gear / RPM
- Time delta, lap progress
- Ambient weather signals

### Derived Features
- Longitudinal acceleration (`dv/dt`)
- Lateral acceleration / cornering intensity proxy
- Braking power proxy (`|a_long| * speed`)
- Slip/energy surrogate terms from throttle + accel + speed
- Rolling-window aggregates (mean/std over short horizons)

### Optional Context Features
- Tire compound (categorical)
- Session type (FP/Quali/Race)
- Track ID encoding
- Driver/car one-hot or embedding
- Pit-stop/tire-change event indicators as pressure-state proxy
- Brake-zone regime features as brake-bias sensitivity proxy

## 6. Physics-Informed Formulation
## 6.1 Thermodynamic Prior (Simplified)
Use a lumped thermal balance prior:

`dT/dt = (Q_in - Q_out) / C_eff`

Where:
- `Q_in` = mechanical work/friction-heating surrogate from telemetry
- `Q_out` = convective/radiative cooling surrogate (dependent on speed and ambient conditions)
- `C_eff` = effective thermal capacity (learned or calibrated scalar/parameter)

## 6.2 Constraint Strategy
Apply two layers of physics enforcement:

1. **Soft constraints in loss (recommended baseline)**
   - Penalize sign/magnitude violations vs expected `dT/dt`
   - Penalize excessive short-time oscillations beyond plausible thermal inertia

2. **Optional hard constraints**
   - Clamp predicted temperature range to plausible physical bounds
   - Monotonic tendency constraints under extreme heating/cooling regimes

## 6.3 Total Loss
`L_total = L_data + λ_phys * L_phys + λ_smooth * L_smooth + λ_bound * L_bound`

- `L_data`: MAE/MSE on `ΔT_core(t→t+5s)` (primary)
- `L_phys`: residual vs simplified thermal ODE trend
- `L_smooth`: temporal smoothness penalty
- `L_bound`: bound penalty for impossible temperature values

Tune `λ_*` via validation sweeps.

## 7. Model Architecture Options

Implementation backend note:
- Preferred on Apple Silicon: **MLX** (`mlx`, `mlx.nn`, `mlx.optimizers`)
- Alternative backend: PyTorch if cross-platform parity is required

## 7.1 Baseline 1: LSTM
- Input: lookback window (e.g., 5–20 seconds)
- 1–3 LSTM layers + dropout
- Dense head for 5-second-ahead delta regression
- Pros: robust, low complexity
- Cons: may underperform on long-range dependencies

## 7.2 Baseline 2: Temporal Transformer
- Input embedding for multivariate sequences
- Positional encoding + attention blocks
- Regression head for `ΔT_core(t→t+5s)`
- Pros: better long-context modeling
- Cons: more data/compute hungry

## 7.3 Physics-Informed Variant
Either architecture + custom physics-informed loss and/or constrained output head.

## 8. Training and Evaluation Plan
## 8.1 Data Split Strategy
Avoid leakage with grouped splits:
- Train/val/test by event or session (not random row split)
- Additional cross-track holdout for generalization check

## 8.2 Metrics
Primary:
- MAE on delta (`deg C / 5s`)
- RMSE on delta (`deg C / 5s`)
- Sign accuracy of `ΔT` (heating vs cooling direction)

Physics validity metrics:
- Violation rate of expected `dT/dt` sign under defined high-heating regimes
- Out-of-bounds prediction rate
- Temporal smoothness/jerk score of predicted temperature

Operational metrics:
- Performance in critical windows (braking zones, corner exits, long straights)

## 8.3 Baseline Comparison Matrix
Compare at minimum:
- Naive zero-delta baseline (`ΔT(t→t+5)=0`)
- Black-box LSTM/Transformer
- Physics-informed LSTM/Transformer

Success criterion:
- Physics-informed model reduces physical violation rate materially **without** unacceptable MAE/RMSE degradation.

## 9. Experiment Design
## 9.1 Ablation Studies
- Remove `L_phys`, keep others
- Vary `λ_phys`
- Remove weather features
- Vary lookback length
- LSTM vs Transformer under same split

## 9.2 Stress Tests
- High-energy segments (aggressive acceleration/braking)
- Low-temp/high-wind sessions
- Cross-event generalization

## 10. Implementation Architecture
## 10.1 Proposed Repository Structure
```text
PITNN/
  data/
    raw/
    processed/
  notebooks/
  src/
    data/
      fastf1_loader.py
      weather_loader.py
      align.py
      features.py
      splits.py
    models/
      lstm.py
      transformer.py
      losses.py
      physics.py
    train/
      train.py
      evaluate.py
      sweep.py
    utils/
      config.py
      logging.py
      metrics.py
  configs/
    baseline_lstm.yaml
    baseline_transformer.yaml
    pinn_lstm.yaml
    pinn_transformer.yaml
  reports/
    figures/
    results/
  README.md
```

## 10.2 Core Modules
- `fastf1_loader.py`: session ingestion + telemetry extraction
- `weather_loader.py`: Open-Meteo pull + caching
- `align.py`: timestamp synchronization
- `features.py`: derived features and normalization
- `physics.py`: thermal priors, residual computations
- `losses.py`: composite objective
- `train.py`: model training loop, checkpoints
- `evaluate.py`: metric computation + plots + violation analytics

## 11. Detailed Implementation Plan (Checklist)
## Phase 0: Project Setup
- [x] Create repository scaffold (`src`, `configs`, `data`, `reports`)
- [x] Pin Python/package versions (`requirements.txt` or `pyproject.toml`)
- [x] Add reproducibility controls (global seed, deterministic flags)
- [x] Add config management (YAML/Hydra/argparse)

## Phase 1: Data Ingestion
- [ ] Implement FastF1 session downloader and local cache
- [ ] Implement telemetry extractor for selected channels
- [ ] Implement Open-Meteo client with timestamped weather fetch
- [ ] Store raw datasets with consistent schema/version tags

## Phase 2: Data Processing
- [ ] Resample all streams to fixed timestep
- [ ] Align telemetry and weather by timestamp
- [ ] Add missing-value handling + mask features
- [ ] Segment into training sequences (lookback, horizon=5s)
- [ ] Generate train/val/test grouped splits by event/session

## Phase 3: Feature Engineering
- [ ] Implement derived acceleration and energy proxy features
- [ ] Add rolling statistics features
- [ ] Encode categorical context (compound, track, session)
- [ ] Normalize/standardize features with train-only statistics

## Phase 4: Baseline Models
- [ ] Implement persistence baseline
- [ ] Implement LSTM regression model
- [ ] Implement Transformer regression model
- [ ] Train and log baseline metrics

## Phase 5: Physics-Informed Components
- [ ] Implement simplified thermal balance residual function
- [ ] Implement composite physics-informed loss (`L_total`)
- [ ] Add optional bound constraints and monotonic heuristics
- [ ] Validate that gradients and loss scaling are stable

## Phase 6: Training Pipeline
- [ ] Add configurable training loop for all model variants
- [ ] Add early stopping, checkpointing, and LR scheduling
- [ ] Add experiment tracking (CSV/MLflow/W&B optional)
- [ ] Run hyperparameter sweep for `λ_phys`, lookback, model size

## Phase 7: Evaluation and Analysis
- [ ] Compute delta MAE/RMSE and sign accuracy across splits
- [ ] Compute physics violation rates and out-of-bound rates
- [ ] Produce segment-level analysis (corners, straights, braking)
- [ ] Create comparative plots for baseline vs physics-informed outputs

## Phase 8: Documentation and Delivery
- [ ] Write methodology and limitations section clearly
- [ ] Document all assumptions and unknown confounders
- [ ] Provide reproducible run commands
- [ ] Prepare final report slides/figures for portfolio/interviews

## 12. Acceptance Criteria
- [ ] End-to-end pipeline runs from raw ingestion to evaluation
- [ ] At least one baseline and one physics-informed model trained successfully
- [ ] Physics-informed model shows reduced invalid-behavior rate and robust delta-direction prediction
- [ ] Results include both error metrics and physics validity metrics
- [ ] Limitations regarding missing aero/suspension/pressure/brake-bias variables are explicitly documented

## 13. Risks and Mitigations
- **Risk:** Noisy/incomplete labels for tire core temperature
  - **Mitigation:** Add label quality filters and confidence tagging
- **Risk:** Over-constraining physics terms harms predictive accuracy
  - **Mitigation:** Tune `λ_phys`; report Pareto curve (error vs validity)
- **Risk:** Data leakage in random splits
  - **Mitigation:** Strict grouped split by session/event
- **Risk:** Poor transfer across tracks/cars
  - **Mitigation:** Include track/car context and holdout evaluation

## 14. Suggested Initial Milestones (2–3 weeks)
- [ ] Week 1: Data ingestion, alignment, baseline persistence + LSTM
- [ ] Week 2: Physics-informed loss integration + ablations
- [ ] Week 3: Evaluation pack, plots, and final write-up

## 15. Minimal Reproducible Runbook (Target)
```bash
# 1) Build dataset
python -m src.data.fastf1_loader --season 2024 --sessions race
python -m src.data.weather_loader --tracks all
python -m src.data.align --config configs/data.yaml

# 2) Train black-box baseline
python -m src.train.train --config configs/baseline_lstm.yaml

# 3) Train physics-informed model
python -m src.train.train --config configs/pinn_lstm.yaml

# 4) Evaluate and compare
python -m src.train.evaluate --run-id baseline_lstm
python -m src.train.evaluate --run-id pinn_lstm
```

## 16. Portfolio/Interview Positioning
Key message:
- This project demonstrates practical motorsport ML maturity: model performance is improved while respecting thermodynamic constraints, with transparent limitations due to missing aero/suspension/pressure/brake-bias observability.

One-line framing:
- "I built a physics-informed sequence model that predicts 5-second tire temperature rise (`ΔT`) and explicitly penalizes thermodynamically implausible behavior."
