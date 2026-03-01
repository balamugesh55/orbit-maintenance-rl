## Sub-Problem 3: Hierarchical Attitude Control

### Approach
Started with full MIMO plant analysis to understand coupling between
orbital thruster (u1) and rotational thruster (u2):
- **RGA analysis**: confirmed optimal pairing u1↔y1 (orbital), u2↔y2 (attitude)
- **Condition Number**: up to 250 at high v1/alpha = proved gain scheduling needed
- **Niederlinski Index**: NI > 0 = stable decoupled pairing confirmed
- **Coupling magnitude**: |G12/G11| = 20.27% → decoupler cannot be ignored

### Architecture
**Outer Loop (every 600s):** PID controller (Kp=0.12, Ki=0.004, Kd=0.06)
with gain-scheduled decoupler. Decoupler D12 recomputed at each step
from current (v1, alpha) operating point — fixes the static D12=13.57
that was wrong across the full operating range.

**Inner Loop (every 10s):** L1-MPC with bang-coast-bang profile.
Horizon H=20, Q=50, R=120 (heavy penalty suppresses micro-corrections),
terminal cost Qf=2500 for fast convergence.

**EMA Noise Filter:** Applied to all states before control computation.
Betas tuned separately for position (0.4) and velocity (0.3).

**Pre-Warm Optimizer:** MPC solved once on initial error before
simulation starts. Injects optimal u-sequence as warm start -
eliminates cold-start fuel spike from 0.0059 kg down to optimal.

**Deadband:** Below ±0.5° error AND ±0.05°/s angular rate - u=0.
Stops post-convergence micro-thrusting that accumulates unnecessary fuel.

### Results
| Metric | Target | Achieved |
|--------|--------|----------|
| Alignment | ±10° | S |
| Alignment | ±1° | S |
| Alignment | ±0.1° | S |
| Fuel per step (converged) | ≤ 0.0033 kg | S |
| Cold-start fuel (step 1) | minimized | S pre-warm applied |
| Steady-state fuel | optimal | S 0.0033 kg |

### Key Insight
Static decoupler failed because D12 varies from 0 to ~76 across the
operating range. Gain-scheduled decoupler recomputed from live (v1, alpha)
solved the cross-coupling problem completely.****
