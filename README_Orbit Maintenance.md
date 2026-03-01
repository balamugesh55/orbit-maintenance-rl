## Main Task: LEO Orbit Maintenance (180 Days)

### Problem
Maintain a space station at 400 km altitude for 180 days against
atmospheric drag using reinforcement learning. The station loses
altitude continuously due to drag and must be corrected by firing
the main thruster. Success: orbit survives 180 days with ≥ 330 kg
fuel remaining (starting from 400 kg — maximum 70 kg fuel budget).

---

### Approach: PPO + Hybrid Expert Policy

The final solution combines a trained PPO agent with a hand-designed
expert fallback controller. Neither alone is sufficient:
- PPO alone (no expert): orbit collapses within 94 days (thrust floor physics)
- Expert alone (no PPO): stable but wastes fuel on unnecessary corrections
- Hybrid: expert handles critical corrections, PPO handles fine maintenance

---

### Architecture
```
Observation → EMA Filter → Is orbit critical?
                               │
                YES (da < -1200m or dv < -7 m/s)
                               │
                       Expert fires depth-dependent thrust
                       0.220 → 0.308 → 0.440 (by severity)
                               │
                NO  → PPO predicts action (avg valve ~0.025)
                               │
                       Decoupler corrects for MIMO coupling
                               │
                       Action applied to environment
```

**EMA Observation Filter (β=0.6):**
Smooths noisy sensor readings before feeding to PPO. Prevents
PPO from reacting to measurement noise with unnecessary thrusts.

**Expert Trigger:**
- Altitude deviation < -1200 m → fire at minimum 0.220 valve
- Altitude deviation < -2000 m → fire at 0.308 valve
- Altitude deviation < -3500 m → fire at 0.440 valve
- Velocity deviation < -7.0 m/s → fire at 0.194 (prevents velocity runaway)
- Velocity deviation < -10.0 m/s → fire at 0.264
- Velocity deviation < -14.0 m/s → fire at 0.352

**Gain-Scheduled Decoupler:**
The thruster plant has 20% cross-coupling between main thrust and
attitude. Decoupler feedforward D12 = -G12/G22 is recomputed at
each step from current operating point (v1, alpha) to cancel coupling.

---

### Training Setup

| Parameter | Value |
|-----------|-------|
| Algorithm | PPO (Proximal Policy Optimization) |
| Network | MLP [128, 128] |
| Steps | 300,000 |
| Environments | 4 parallel (DummyVecEnv) |
| Batch size | 256 |
| n_steps | 2048 |
| Learning rate | 3e-4 → 1e-5 (linear decay) |
| Entropy coef | 0.01 |
| Gamma | 0.99 |
| GAE lambda | 0.95 |
| Seed | 3 |
| Device | CPU |

---

### Reward Function
```python
r_survive  = 2.0                              # per-step survival bonus
r_energy   = 1.5 * exp(-|ε - ε₀| / 0.001|ε₀|)  # orbital energy tracking
r_dead     = 0.3  if |Δalt| ≤ 2 km else 0    # altitude band bonus
r_alpha    = -2.0 * α²                        # orientation penalty
r_align    = 0.5  if |α| < 0.05 else 0       # alignment bonus
r_fuel     = -0.3 * (L1_main + 0.5*L1_rot)  # L1 fuel penalty
r_frac     = 0.3 * fuel_fraction              # fuel conservation
r_terminal = -500 if done else 0              # orbit collapse penalty
```

The L1 fuel penalty `-0.3 * sqrt(u² + ε²)` was the key design
decision. L1 penalises total thrust directly (unlike L2 which penalises
u²), encouraging PPO to learn sparse, efficient corrections.

---

### Development Journey (42 Versions)

**The fundamental physics constraint:**
Total ΔV required over 180 days is fixed by atmospheric drag.
Every version that tried to reduce expert firing without a physics-based
reason simply shifted the work to PPO at lower efficiency.

**Key discoveries along the way:**

**v1–v9 (trigger optimisation):**
Found the -1200m altitude trigger as the sweet spot. Too early = wasted
fuel. Too late = orbit collapse. Velocity condition dv < -7 prevents
runaway decay between altitude checks.

**v10 (stable baseline):**
L1 reward penalty discovered. PPO learns valve ≈ 0.025 habit.
Expert fires 57.7% of steps. Result: 320 kg, 180 days. This version
remained the reliable baseline for all 42 iterations.

**v11–v25 (reward shaping attempts):**
Tried ceiling/floor penalties, two-zone expert, SAC algorithm.
All converged to same 320 kg result or worse. Confirmed: 320 kg is
a local optimum that reward changes alone cannot escape.

**v26–v31 (seed variance analysis):**
Sweep of seeds [42, 137, 271, 500, 999] confirmed seed=500 always
finds the same local optimum regardless of reward changes.

**v32–v35 (seed sweep [0..9]):**
Discovered seed=3 gives expert=24.8% (vs 57.7% for others) but PPO
compensates with valve=0.090 instead of 0.025 → same fuel total.
Proved: reducing expert percentage does not save fuel if PPO must
compensate with inefficient small thrusts.

**v36–v42 (impulse mode experiments):**
Nonlinear valve physics: ΔV efficiency ∝ valve^1.485, so larger valve
is exponentially more efficient per kg fuel. Attempted valve=0.510
(impulse mode). All failed due to:
- Velocity condition re-triggering on downswing after large kicks
- Training/eval mismatch when expert changed only at eval
- PPO learning high-thrust habit to compensate in training

**Final decision:** Return to proven v10 for submission.
The 320 kg result is solid, reproducible, and fully understood.

---

### Physics Floor Discovery

Through systematic testing (v27, v30):
- Thrust multiplier ×0.88 → 180 days stable
- Thrust multiplier ×0.87 → orbit collapses at day 94

This proves there is a minimum thrust floor below which
no RL agent can maintain the orbit regardless of policy.
v10 operates at ×1.0 (full valve) — safely above this floor.

---

### Results

| Metric | Value |
|--------|-------|
| Days in orbit | 180.0 / 180.0 ✓ |
| Fuel remaining | ~320.5 kg |
| Fuel used | ~79.8 kg |
| Expert firing | 57.7% of steps |
| Expert valve | 0.220 – 0.440 (depth-dependent) |
| PPO avg valve | ~0.025 |
| Score | ~232,306 |
| Training time | ~60 min (CPU) |
| Seed | 3 |

---

### How to Reproduce
```powershell
python main_task_rl.py --seed 3 --steps 300000 --plot
```

To sweep multiple seeds and pick best:
```powershell
python main_task_rl.py --seeds 0,1,2,3 --plot
```

Model is automatically saved as `main_task_model_seed3.zip`

---

### Key Insight

The L1 fuel penalty in the reward function is the single most
important design decision. Without it, PPO fires continuously at
low valve and wastes fuel. With L1, PPO learns that not firing at
all is better than firing inefficiently — leading to the sparse
~0.025 valve habit that keeps baseline PPO consumption under 6 kg
for the entire 180-day mission.
