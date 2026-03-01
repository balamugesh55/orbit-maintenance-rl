# LEO Space Station Orbit Maintenance — Reinforcement Learning
**Course:** Computational Methods in Space Engineering  
**University:** University of Bremen — Space Engineering  
**Year:** 2025-2026

---

## Project Overview

This repository contains solutions for the LEO (Low Earth Orbit) space station 
keeping assignment. The task involves maintaining a space station at 400 km altitude 
for 180 days using reinforcement learning and classical control methods.

The project is split into four parts:

**Main Task — Orbit Maintenance (180 days)**  
A PPO (Proximal Policy Optimization) agent with a hybrid expert policy maintains 
the station's orbital altitude against atmospheric drag. The expert fires the main 
thruster when altitude drops below -1200m, while the RL agent handles fine 
corrections. Best result: 180 days stable orbit, 320+ kg fuel remaining.

**Sub-Problem 1 — System Identification (Curve Fitting)**  
Identifies the physical parameters of the station's thrusters and drag model 
from noisy sensor data. Uses Differential Evolution + curve_fit polish to fit 
power-law thrust curves and a sinusoidal drag model. Accuracy: 97-99% R².

**Sub-Problem 2 — Attitude Control (L1-MPC)**  
Replaces the default PID controller with a minimum-fuel Model Predictive 
Controller. Uses L1 cost (|u| instead of u²) to produce bang-coast-bang thrust 
profiles. Result: 0.00° alignment error, fuel ≤ 0.0033 kg per 600s step.

**Sub-Problem 3 — Hierarchical Control**  
Full MIMO control architecture: PID outer loop (600s) + L1-MPC inner loop (10s) 
+ gain-scheduled decoupler + EMA noise filter + optimizer pre-warming. 
MIMO RGA analysis confirms optimal pairing. Result: ±0.1° alignment, 
steady-state fuel ≤ 0.0033 kg/step.

---

## Repository Structure
```
orbit-maintenance-rl/
│
├── main_task_rl.py              # Main task — PPO hybrid expert
├── main_task_model_seed1.zip    # Trained PPO model 
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
Navigate to the project folder and install dependencies:
```powershell
cd D:\comp_eng\project_handout
pip install numpy scipy matplotlib stable-baselines3 gymnasium torch
```

### 3. Set Project Path
In each script, update the PROJECT_PATH variable to match your local path:
```python
PROJECT_PATH = r"D:\comp_eng\project_handout"
```

### 4. Run in Spyder
All scripts were developed and tested in **Spyder IDE**.  
Open Spyder, set the working directory to your project folder, and run each file.


---

## Results Summary

| Task | Metric | Result | Target | Status |
|------|--------|--------|--------|--------|
| Main Task | Days in orbit | 180.0 days | 180 days | ✓ |
| Main Task | Fuel remaining | 320+ kg | ≥ 330 kg | ~ |
| Sub-Problem 1 | Main thrust R² | 99.88% | high | ✓ |
| Sub-Problem 1 | Drag factor R² | 97.33% | high | ✓ |
| Sub-Problem 2 | Alignment error | 0.01° | ≤ 10° | ✓ |
| Sub-Problem 2 | Fuel per step | 0.0033 kg | ≤ 0.0033 kg | ✓ |
| Sub-Problem 3 | Alignment error | < 0.1° | ≤ 10° | ✓ |
| Sub-Problem 3 | Fuel per step | ≤ 0.0033 kg | ≤ 0.0033 kg | ✓ |

---

## How to Reproduce Results

**Main Task:**
```powershell
python main_task_rl.py --seed 1 --steps 300000 --plot
```

**Sub-Problem 1:**
```powershell
python subproblem1_advanced.py
```

**Sub-Problem 2:**
```powershell
python subproblem2_advanced.py
```

**Sub-Problem 3:**
```powershell
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
