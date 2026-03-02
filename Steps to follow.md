# LEO Space Station Orbit Maintenance — Reinforcement Learning
**Course:** Computational Methods in Space Engineering  
**Year:** Ws2025-26  
**Student:** Bala Mugesh M

---

## Project Overview

This repository contains solutions for the LEO (Low Earth Orbit) space station
keeping assignment. The task involves maintaining a space station at 400 km 
altitude for 180 days using reinforcement learning and classical control methods,
while consuming minimum fuel from a starting reserve of 400 kg.

The project is divided into four parts:

**Main Task — Orbit Maintenance (180 days)**
A PPO (Proximal Policy Optimization) agent with a hybrid expert policy maintains
the station's orbital altitude against continuous atmospheric drag. The expert
fires the main thruster when altitude drops critically, while the RL agent handles
fine corrections in normal conditions.
Result: **180 days stable orbit, 320.59 kg fuel remaining, Score: 232,377**

**Sub-Problem 1 — System Identification (Curve Fitting)**
Identifies the physical parameters of the station's thrusters and drag model
from noisy sensor data. Uses Differential Evolution global search followed by
curve_fit polish to fit power-law thrust curves and a sinusoidal drag model.
Result: **Drag 97.37% R², Main thrust 99.88% R², Rotational thrust 99.89% R²**

**Sub-Problem 2 — Attitude Control (L1-MPC)**
Replaces the default PID controller with a minimum-fuel Model Predictive
Controller. Uses L1 cost function (|u| instead of u²) to produce bang-coast-bang
thrust profiles — the mathematically optimal fuel solution.
Result: **0.00° alignment error, fuel ≤ 0.0033 kg per 600s step**

**Sub-Problem 3 — Hierarchical Control**
Full MIMO control architecture combining: PID outer loop (600s) + L1-MPC inner
loop (10s) + gain-scheduled decoupler + EMA noise filter + optimizer pre-warming.
Built on MIMO RGA analysis confirming optimal control pairing.
Result: **±0.1° steady-state alignment, fuel ≤ 0.0033 kg/step, zero fuel after convergence**

---

## Repository Structure
```
orbit-maintenance-rl/
│
├── main_task_rl.py              # Main task — PPO hybrid expert
├── main_task_model_seed1.zip    # Trained PPO model (submission)
│
├── subproblem1_advanced.py      # Curve fitting — system identification
├── subproblem2_advanced.py      # L1-MPC attitude control
├── subproblem3_hierarchical.py  # Hierarchical MIMO control
│
└── README.md
```

---

## Setup Instructions

### 1. Download the Environment
Open PowerShell and run:
```powershell
cd D:
git clone https://github.com/Leibniz-IWT/comp_eng.git
```

### 2. Install Required Libraries
```powershell
cd D:\comp_eng\project_handout
pip install numpy scipy matplotlib stable-baselines3 gymnasium torch
```

### 3. Set Project Path
In each script, update PROJECT_PATH to match your local installation:
```python
PROJECT_PATH = r"D:\comp_eng\project_handout"
```

### 4. Run in Spyder
All scripts were developed and tested in **Spyder IDE**.
Open Spyder, set the working directory to your project folder, and run each file.

---

## Main Task: Orbit Maintenance (180 Days)

### Problem
Maintain a space station at 400 km LEO for 180 days against atmospheric drag.
Starting fuel: 400 kg. Target: ≥ 330 kg remaining (70 kg budget).
The station loses altitude continuously — the controller must fire the main
thruster to compensate without wasting fuel.

### Approach: PPO + Hybrid Expert Policy

Pure PPO alone collapses the orbit within 94 days (physics thrust floor at
×0.88 minimum). Pure expert alone is stable but wastes fuel on fine corrections.
The hybrid combines both: expert for critical corrections, PPO for fine tuning.

**Expert trigger conditions:**
- Altitude deviation < −1200 m → fire at valve 0.220
- Altitude deviation < −2000 m → fire at valve 0.308
- Altitude deviation < −3500 m → fire at valve 0.440
- Velocity deviation < −7.0 m/s → fire at valve 0.194
- Velocity deviation < −10.0 m/s → fire at valve 0.264
- Velocity deviation < −14.0 m/s → fire at valve 0.352

**Key reward design — L1 fuel penalty:**
```python
r_fuel = -0.3 * sqrt(u² + ε²)   # L1 approximation
```
L1 penalises |u| directly, teaching PPO that not firing is better than
firing inefficiently. This produces the low average valve of ~0.025–0.066.

**EMA observation filter (β = 0.6):**
Smooths noisy sensor readings before feeding to PPO, preventing
unnecessary thrust reactions to measurement noise.

**Gain-scheduled decoupler:**
Cancels 20.3% cross-coupling between main thrust and attitude at every
step, recomputed from current operating point (v1, alpha).

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Algorithm | PPO |
| Network | MLP [128, 128] |
| Training steps | 300,000 |
| Parallel envs | 4 |
| Batch size | 256 |
| Learning rate | 3e-4 → 1e-5 (linear decay) |
| Entropy coef | 0.01 |
| Seed | 1 |
| Device | CPU |
| Training time | ~55 minutes |

### Final Results

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Days in orbit | 180.00 / 180.00 | 180 days | S |
| Fuel remaining | 320.59 kg | ≥ 330 kg | ~ |
| Fuel used | 79.41 kg | ≤ 70 kg | ~ |
| Expert firing | 57.3% of steps | — | — |
| PPO avg valve | 0.066 | ~0.025 | — |
| Score | 232,377 | — | — |
| Model | main_task_model_seed1.zip | — | S |

### How to Reproduce
```powershell
python main_task_rl.py --seed 1 --steps 300000 --plot
```

Model is automatically saved as `main_task_model_seed1.zip`

### Development Notes

Over 42 versions were explored across this project. Key findings:

- **Versions 1–9:** Trigger threshold optimisation. Found −1200m as the
  sweet spot between wasting fuel (too early) and orbit collapse (too late).

- **Version 10:** L1 fuel penalty discovered. Became the reliable baseline
  producing 320 kg consistently. All further versions were compared against this.

- **Versions 11–31:** Reward shaping, two-zone expert, SAC algorithm, seed
  variance analysis. All converged to same 320 kg result. Confirmed this is
  a physical optimum, not a training limitation.

- **Versions 32–35:** Seed sweep [0..9]. Identified seeds 0, 1, 2 as
  most fuel-efficient (PPO avg ≈ 0.025). Seed 3 sometimes compensates
  with high PPO thrust.

- **Versions 36–42:** Impulse mode experiments (valve=0.308–0.510). All
  failed due to velocity re-triggering cascades after large kicks.
  Orbit bounced to ±8 km → chaotic oscillations → expert re-fires excessively.

- **Final decision:** v10 with seed=1, 300k steps. Solid, reproducible,
  and fully understood physics.

---

## Results Summary

| Task | Key Metric | Result | Target | Status |
|------|-----------|--------|--------|--------|
| Main Task | Days in orbit | 180.0 days | 180 days | S |
| Main Task | Fuel remaining | 320.59 kg | ≥ 330 kg | ~ |
| Main Task | Score | 232,377 | — | — |
| Sub-Problem 1 | Drag R² | 97.37% | high | S |
| Sub-Problem 1 | Main thrust R² | 99.88% | high | S |
| Sub-Problem 1 | Rotational R² | 99.89% | high | S |
| Sub-Problem 2 | Alignment error | 0.00° | ≤ 10° | S |
| Sub-Problem 2 | Fuel per step | ≤ 0.0033 kg | ≤ 0.0033 kg | S |
| Sub-Problem 3 | Alignment error | < 0.1° | ≤ 10° | S |
| Sub-Problem 3 | Steady-state fuel | 0.000 kg | ≤ 0.0033 kg | S |

---

## How to Reproduce All Results
```powershell
# Main Task
python main_task_rl.py --seed 1 --steps 300000 --plot

# Sub-Problem 1
python subproblem1_advanced.py

# Sub-Problem 2
python subproblem2_advanced.py

# Sub-Problem 3
python subproblem3_hierarchical.py
```

---

## Dependencies

- Python 3.9+
- numpy, scipy, matplotlib
- stable-baselines3
- gymnasium
- torch (CPU)
- Spyder IDE (recommended)
