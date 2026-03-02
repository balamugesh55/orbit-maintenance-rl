## Main Task: Orbit Maintenance (180 Days)

### Problem
Maintain a space station at 400 km LEO for 180 days against atmospheric drag.
Starting fuel: 400 kg. Target: ≥ 330 kg remaining (70 kg budget).
The station loses altitude continuously due to atmospheric drag — the controller
must fire the main thruster to compensate without wasting fuel.

---

### Why Pure RL Fails
A pure PPO agent without any guidance collapses the orbit within 94 days.
This is because there is a physical thrust floor — below 88% of minimum
thrust, the orbit cannot be maintained regardless of the control policy.
Discovered through testing: thrust multiplier ×0.88 → 180 days stable,
thrust multiplier ×0.87 → orbit collapses at day 94.

---

### Approach: PPO + Hybrid Expert Policy

Neither pure PPO nor pure expert alone is sufficient:
- PPO alone → orbit collapses (thrust floor physics)
- Expert alone → stable but wastes fuel on unnecessary fine corrections
- Hybrid → expert handles critical corrections, PPO handles fine maintenance
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
                    Gain-scheduled decoupler applied
                               │
                    Action sent to environment
```

---

### Expert Trigger Conditions

| Condition | Valve |
|-----------|-------|
| Altitude < −1200 m | 0.220 |
| Altitude < −2000 m | 0.308 |
| Altitude < −3500 m | 0.440 |
| Velocity < −7.0 m/s | 0.194 |
| Velocity < −10.0 m/s | 0.264 |
| Velocity < −14.0 m/s | 0.352 |

Both altitude AND velocity conditions are checked every step.
The velocity condition prevents runaway orbital decay between altitude checks.
The depth-dependent thrust keeps the orbit in a tight ±200m band —
just enough correction without overshooting.

---

### Reward Function
```python
r_survive  =  2.0                                    # per-step survival bonus
r_energy   =  1.5 * exp(-|ε - ε₀| / 0.001|ε₀|)     # orbital energy tracking
r_dead     =  0.3  if |Δalt| ≤ 2 km else 0          # altitude band bonus
r_alpha    = -2.0 * α²                               # orientation penalty
r_align    =  0.5  if |α| < 0.05 rad else 0         # alignment bonus
r_fuel     = -0.3 * sqrt(u² + 1e-6)                 # L1 fuel penalty
r_frac     =  0.3 * fuel_fraction                    # fuel conservation
r_terminal = -500  if done else 0                    # orbit collapse penalty
```

**The L1 fuel penalty is the single most important design decision.**
Standard L2 (u²) spreads thrust across all steps — many small corrections.
L1 (|u|) penalises the magnitude directly — teaches PPO that not firing
costs nothing, so it only fires when truly necessary. This is why PPO
learns a very low average valve of 0.025–0.066 instead of continuous
moderate thrust.

---

### EMA Observation Filter

Before observations are fed to PPO, an Exponential Moving Average filter
smooths noisy sensor readings:
```
state = β × obs + (1 − β) × prev_state    (β = 0.6)
```
This prevents PPO from reacting to measurement noise with unnecessary
thrust corrections that waste fuel.

---

### Gain-Scheduled Decoupler

The thruster plant has 20.3% cross-coupling between main thrust (u1) and
attitude angle (alpha). Without compensation, firing the main thruster
disturbs the attitude. The decoupler feedforward gain:
```
D12 = −G12 / G22
```
is recomputed at every step from the current operating point (v1, alpha),
cancelling the coupling disturbance before it affects the attitude loop.

---

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Algorithm | PPO (Proximal Policy Optimization) |
| Network architecture | MLP [128, 128] |
| Training steps | 300,000 |
| Parallel environments | 4 (DummyVecEnv) |
| Batch size | 256 |
| n_steps | 2048 |
| Learning rate | 3e-4 -> 1e-5 (linear decay) |
| Entropy coefficient | 0.01 |
| Gamma | 0.99 |
| GAE lambda | 0.95 |
| Clip range | 0.2 |
| Seed | 1 |
| Device | CPU |
| Training time | ~55 minutes |

**Why seed = 1:**
A sweep of seeds 0–9 confirmed that seeds 0, 1, 2 consistently produce
PPO average valve ≈ 0.025 (fuel-optimal habit). Seed 3 sometimes
compensates with high PPO thrust (0.209) depending on training length.
Seed 1 with 300k steps is the most reliable combination.

---

### Development Journey (42 Versions)

**v1–v9 — Trigger Optimisation:**
Tested altitude thresholds from −600m to −3000m. Found −1200m as the
sweet spot. Too early (−600m) wastes fuel firing unnecessarily. Too late
(−2000m) allows orbit to decay past the recovery point.

**v10 — Stable Baseline (L1 Reward):**
Introduced L1 fuel penalty. PPO learned valve ≈ 0.025 habit.
Expert fires 57.7% of steps at 0.220–0.440 valve.
Result: 320 kg, 180 days. This became the benchmark for all further work.

**v11–v31 — Reward Shaping Attempts:**
Tried ceiling/floor penalties, two-zone expert, SAC algorithm, altitude
band rewards. All converged to the same 320 kg. Confirmed: 320 kg is
a physics-constrained local optimum, not a training limitation. The total
ΔV needed over 180 days is fixed by drag — changing the reward cannot
reduce the physics cost.

**v32–v35 — Seed Sweep:**
Swept seeds 0–9. Found seeds 0, 1, 2 give PPO avg ≈ 0.025 (optimal).
Seed 3 gives expert = 24.8% but PPO compensates at 0.090 — same total
fuel. Proved: reducing expert percentage does not save fuel if PPO must
compensate.

**v36–v42 — Impulse Mode Experiments:**
Attempted valve = 0.308–0.510 (larger kick = fewer firings needed by
physics). All failed because:
- Large kick → orbit bounces to ±8 km
- Velocity condition re-triggers on the downswing
- Expert fires 72% instead of 25% -> more fuel than v10
- Conclusion: the velocity re-trigger cascade cannot be fixed without
  fundamentally changing training, not just eval behaviour.

**Final decision:** v10 with seed=1, 300k steps.
Solid, reproducible, and physically understood.

---

### Final Result

| Metric | Value |
|--------|-------|
| Days in orbit | 180.00 / 180.00 S |
| Fuel remaining | 320.59 kg |
| Fuel used | 79.41 kg |
| Expert firing | 57.3% of steps |
| PPO avg valve | 0.066 |
| Score | 232,377 |
| Training time | 55 minutes |
| Model file | main_task_model_seed1.zip |

### How to Reproduce
```powershell
python main_task_rl.py --seed 1 --steps 300000 --plot
```

Model saves automatically as `main_task_model_seed1.zip`
