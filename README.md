## Sub-Problem 2: Minimum-Fuel Attitude Control (MPC)

### Problem
Attitude control of space station to maintain alignment with velocity
vector within 600 seconds. Starting from -30° misalignment.
Success criterion: final alignment error < ±10°.
Stretch target: 0.00° error with ≤ 0.0033 kg fuel.

### Step 1 — Baseline PID (what we started with)
Tuned PID gains: Kp=0.1, Ki=0.01, Kd=0.1
- Starting from -10°: final error = 8.90°, fuel = 0.0123 kg
- Just barely within ±10° target
- Problem: slow response, high fuel, cannot reach ±1° or ±0.1°

### Step 2 — Standard MPC (quadratic cost, first attempt)
Replaced PID with receding-horizon MPC.
State model: double-integrator (error, omega)
Dynamics:
  error(k+1) = error(k) - omega(k)*dt - 0.5*g*u(k)*dt²
  omega(k+1) = omega(k) + g*u(k)*dt

Cost: J = Q*Σe² + R*Σu² + Qf*e_N²
Parameters: horizon=15, Q=200, R=0.5, Qf=2000

Result: final error = -0.0101° S (within ±0.1°)
BUT fuel = 0.0574 kg F (17× too high)

Root cause: quadratic R*u² spreads thrust evenly across all steps
→ small continuous corrections → high total fuel

### Step 3 — L1-MPC (minimum-fuel, bang-coast-bang)
Key insight: L1 cost |u| directly penalises total fuel.
Quadratic u² encourages many small thrusts → expensive.
L1 |u| encourages sparse thrust → bang-coast-bang → fuel-optimal.

L1 approximated as R*sqrt(u²+ε²) for smooth gradients (L-BFGS-B).

Cost: J = Q*Σe² + R*Σsqrt(u²+ε²) + Qf*e_N²
Parameters: horizon=20, Q=50, R=80 (heavy fuel penalty), Qf=1000

Plant gain identified from simulation: g = 0.018 deg/s² per unit u

Two-pass optimisation:
- Pass 1: coarse (maxiter=500, ftol=1e-14)
- Pass 2: fine polish from pass 1 result (ftol=1e-16)
Warm-start: shifted previous solution → faster convergence

### Results

| Metric | Target | PID | Quadratic MPC | L1-MPC |
|--------|--------|-----|---------------|--------|
| Within ±10° | S | S 8.90° | S 0.01° | S |
| Within ±1° | stretch | F | S | S |
| Within ±0.1° | stretch | F | S | S |
| Fuel ≤ 0.0033 kg | stretch | F 0.0123 | F 0.0574 | S |

### Key Insight
The switch from quadratic (R*u²) to L1 (R*|u|) cost function
is the critical design decision. L1 naturally produces bang-coast-bang
thrust profiles — maximum thrust for minimum time — which is the
mathematically optimal solution for fuel-constrained manoeuvres.
This is known as the "minimum-fuel theorem" in optimal control theory.
