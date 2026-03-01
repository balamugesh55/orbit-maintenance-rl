# ============================================================
# SUB-PROBLEM 2: Minimum-Fuel MPC - Bala Mugesh
#
# Key insight: quadratic cost R*u² spreads thrust across all steps
#              - high fuel use (0.0574 kg)
#
# Fix: L1 cost R*|u| penalises total fuel DIRECTLY
#      - optimizer naturally finds bang-coast-bang solution
#      - sparse thrust: thrust only when necessary - low fuel
#
# L1 is approximated as R*sqrt(u²+ε) for smooth gradients (L-BFGS-B)
#
# MPC state-space model (degrees, dt=10s):
#   error(k+1) = error(k)  - omega(k)*dt - 0.5*g*u(k)*dt²
#   omega(k+1) = omega(k)  + g*u(k)*dt
#
# g (plant gain) is identified from env: the rotational thruster
# produces an angular acceleration of g rad/s² per unit valve signal.
# ============================================================

import sys
import numpy as np
from scipy.optimize import minimize
from types import MethodType

PROJECT_PATH = r"D:\comp_eng\project_handout"
sys.path.insert(0, PROJECT_PATH)

from station_model._sub_problem import attitude_control_sub_problem
from station_model._environment import OrbitMaintenanceEnv


# ============================================================
# PLANT GAIN — identified from simulation
# ============================================================
# From PID test: Kp=0.1, psi0=-10°, final=-8.90° over 600s
# This means env dynamics use degrees internally.
# g is tuned so MPC model matches actual environment response.
G_PLANT = 0.018      # degrees/s² per unit control signal  (tuned)
DT_SUB  = 10.0       # seconds per sub-step (fixed by problem)
EPS_L1  = 1e-3       # smoothing for L1 approximation: |u| ≈ sqrt(u²+ε²)


# ============================================================
# MINIMUM-FUEL MPC CONTROLLER
# ============================================================

def mpc_min_fuel(self, error, integral_error, derivative,
                 current_psi, current_omega, target_alpha, dt_sub):
    """
    Minimum-fuel receding-horizon MPC.

    Cost: J = Q*sum(e²) + Q_f*e_N²  +  R * sum(sqrt(u²+ε))
                                              ^^^^^^^^^^^
                                         L1 ≈ total fuel (sparse)

    Result: bang-coast-bang — thrust only at start & end of manoeuvre.
    """
    H   = self.horizon
    Q   = self.Q
    Q_f = Q * 20.0           # strong terminal cost: must reach 0° by end
    R   = self.R              # high value → minimise fuel
    u_max = self.u_max
    dt  = dt_sub
    g   = G_PLANT

    e0 = float(error)
    w0 = float(current_omega)

    def cost(u_seq):
        e, w = e0, w0
        J = 0.0
        for k in range(H):
            u_k = u_seq[k]
            J  += Q * e**2
            # L1 fuel penalty (smooth approximation)
            J  += R * np.sqrt(u_k**2 + EPS_L1**2)
            # Propagate dynamics
            e   = e - w * dt - 0.5 * g * u_k * dt**2
            w   = w + g * u_k * dt
        J += Q_f * e**2      # terminal: error must be near 0 at horizon end
        return J

    # Warm-start from previous shifted solution
    if not hasattr(self, '_u_prev') or len(self._u_prev) != H:
        # Cold start: guess bang-coast-bang shape
        u0 = np.zeros(H)
        u0[0]  =  u_max * 0.5    # initial bang
        u0[-1] = -u_max * 0.5    # braking bang
    else:
        u0 = np.append(self._u_prev[1:], 0.0)

    bounds = [(-u_max, u_max)] * H

    # Two-pass optimisation: coarse then fine
    res = minimize(cost, u0, method='L-BFGS-B', bounds=bounds,
                   options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10})
    # Fine polish from result
    res2 = minimize(cost, res.x, method='L-BFGS-B', bounds=bounds,
                    options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-12})

    u_opt = res2.x if res2.fun < res.fun else res.x
    self._u_prev = u_opt

    return float(u_opt[0])


# ============================================================
# HYPERPARAMETER TUNING
# ============================================================

env = OrbitMaintenanceEnv()

# Tune these for best performance:
env.horizon = 35       # 20 steps × 10s = 200s lookahead (full manoeuvre visible)
env.Q       = 50.0     # moderate alignment penalty
env.R       = 99     # HEAVY fuel penalty → L1 minimisation → bang-coast-bang
env.u_max   = 10.0     # maximum control signal

# Bind minimum-fuel MPC
env.control_signal = MethodType(mpc_min_fuel, env)

print("="*62)
print("SUB-PROBLEM 2: MINIMUM-FUEL MPC ATTITUDE CONTROL")
print("="*62)
print(f"  Controller   : L1-MPC (bang-coast-bang)")
print(f"  Horizon      : {env.horizon} steps ({env.horizon * DT_SUB:.0f} s)")
print(f"  Q  (align)   : {env.Q}")
print(f"  R  (fuel/L1) : {env.R}  - heavy fuel penalty")
print(f"  Q_f (terminal): {env.Q * 20.0}")
print(f"  u_max        : {env.u_max}")
print(f"  g_plant      : {G_PLANT} deg/s² per unit u")
print(f"  Cost type    : L1 = sqrt(u²+{EPS_L1}²)  [sparse/fuel-optimal]")
print("-"*62)

# ============================================================
# RUN SIMULATION
# ============================================================

alphas, psis, cum_fuel_rot, total_fuel_rot = attitude_control_sub_problem(
    env,
    alpha_setpoint=0.0,
    current_psi=-30,
    dt_sub=DT_SUB
)

# ============================================================
# RESULTS
# ============================================================

final_error = alphas[-1]

print("\n" + "="*62)
print("RESULTS")
print("="*62)
print(f"  Final alignment error : {final_error:.4f}°")
print(f"  Total fuel used       : {total_fuel_rot:.4f} kg")
print(f"  Within ±10°           : {'S YES' if abs(final_error) <= 10   else 'F NO'}")
print(f"  Within ±1°            : {'S YES' if abs(final_error) <= 1    else 'F NO'}")
print(f"  Within ±0.1°          : {'S YES' if abs(final_error) <= 0.1  else 'F NO'}")
print(f"  Fuel ≤ 0.0033 kg      : {'S YES' if total_fuel_rot  <= 0.0033 else f'F NO  - try increasing R (currently {env.R})'}")
print("="*62)

# ============================================================
# TUNING GUIDANCE (printed if targets not met)
# ============================================================

if total_fuel_rot > 0.0033:
    ratio = total_fuel_rot / 0.0033
    print(f"\n  TUNING HINT: fuel is {ratio:.1f}× too high.")
    print(f"  - Increase R from {env.R} to ~{env.R * ratio:.0f}")
    print(f"  - Or increase horizon from {env.horizon} to {env.horizon + 5}")

if abs(final_error) > 1.0:
    print(f"\n  TUNING HINT: alignment error {final_error:.2f}° too large.")
    print(f"  - Increase Q from {env.Q} to ~{env.Q * 2:.0f}")
    print(f"  - Or increase Q_f multiplier (currently 20×)")

print("\n Sub Task 2 Complete!")

