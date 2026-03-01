# ============================================================
# SUB-PROBLEM 1:  SOLUTION Bala Mugesh
# ============================================================

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, differential_evolution
from scipy.interpolate import interp1d

PROJECT_PATH = r"D:\comp_eng\project_handout"
sys.path.insert(0, PROJECT_PATH)

from station_model._hidden_functions import (
    alphas, noisy_drag,
    valves, noisy_thrust_main, noisy_thrust_rot,
    test_solutions
)

# ============================================================
# HELPER: accuracy metrics
# ============================================================

def accuracy_report(y_true, y_pred, label):
    rmse      = np.sqrt(np.mean((y_true - y_pred)**2))
    mae       = np.mean(np.abs(y_true - y_pred))
    ss_res    = np.sum((y_true - y_pred)**2)
    ss_tot    = np.sum((y_true - np.mean(y_true))**2)
    r2        = 1 - ss_res / ss_tot
    maxe      = np.max(np.abs(y_true - y_pred))
    rel_rmse  = rmse / np.mean(y_true) * 100

    # R²-based accuracy: most reliable for regression (unaffected by scale/zeros)
    r2_accuracy = r2 * 100

    # Median APE: robust to outlier low-value points (use values > 5% of max)
    threshold = 0.05 * np.max(np.abs(y_true))
    valid     = np.abs(y_true) > threshold
    if np.sum(valid) > 5:
        ape      = np.abs((y_true[valid] - y_pred[valid]) / y_true[valid]) * 100
        mdape    = np.median(ape)        # median: not skewed by low-value outliers
        mape     = np.mean(ape)
    else:
        mdape, mape = 0.0, 0.0

    print(f"  RMSE            : {rmse:.6f}  ({rel_rmse:.3f}% of mean value)")
    print(f"  MAE             : {mae:.6f}")
    print(f"  R²              : {r2:.6f}")
    print(f"  Max Error       : {maxe:.6f}")
    print(f"  MdAPE           : {mdape:.4f}%  (median, robust to low-value points)")
    print(f"  * Accuracy (R²) : {r2_accuracy:.4f}%")
    return rmse, r2, r2_accuracy


print("="*70)
print("SUB-PROBLEM 1: ADVANCED SOLUTION")
print("="*70)


# ============================================================
# 1. DRAG FACTOR FITTING
# ============================================================

print("\n1. FITTING DRAG FACTOR (ADVANCED)")
print("-"*70)

# Model 1: standard form  a + k*|sin(α)|
def drag_2p(alpha, a, k):
    return a + k * np.abs(np.sin(alpha))

# Model 2: extended form  a + k*|sin(α)|^p  (extra exponent)
def drag_3p(alpha, a, k, p):
    return a + k * np.power(np.abs(np.sin(alpha)), p)

# --- Fit 2-parameter model ---
best_params_2p = None
best_err_2p    = float('inf')
for p0 in [[1.0, 0.5], [1.0, 1.0], [1.2, 0.8], [0.8, 1.2], [0.9, 0.9]]:
    try:
        params, _ = curve_fit(drag_2p, alphas, noisy_drag,
                              p0=p0, maxfev=50000)
        err = np.sqrt(np.mean((noisy_drag - drag_2p(alphas, *params))**2))
        if err < best_err_2p:
            best_err_2p    = err
            best_params_2p = params
    except Exception:
        continue

# --- Fit 3-parameter model with DE global search ---
de_drag = differential_evolution(
    lambda p: np.sum((noisy_drag - drag_3p(alphas, *p))**2),
    bounds=[(0.5, 1.5), (0.5, 2.0), (0.3, 3.0)],
    seed=42, maxiter=3000, tol=1e-14, polish=True
)
a3, k3, p3 = de_drag.x
# Polish with curve_fit
try:
    params_3p, _ = curve_fit(drag_3p, alphas, noisy_drag,
                             p0=[a3, k3, p3], maxfev=100000)
    err_3p = np.sqrt(np.mean((noisy_drag - drag_3p(alphas, *params_3p))**2))
except Exception:
    params_3p = de_drag.x
    err_3p = np.sqrt(np.mean((noisy_drag - drag_3p(alphas, *params_3p))**2))

# --- Pick best model ---
print(f"  2-param fit: a={best_params_2p[0]:.6f}, k={best_params_2p[1]:.6f}  RMSE={best_err_2p:.6f}")
print(f"  3-param fit: a={params_3p[0]:.6f}, k={params_3p[1]:.6f}, p={params_3p[2]:.6f}  RMSE={err_3p:.6f}")

if err_3p < best_err_2p:
    print(f"  → 3-param model wins (p≠1 confirms non-unit exponent)")
    a_fit, k_fit = params_3p[0], params_3p[1]
    p_fit = params_3p[2]
    drag_pred_alphas = drag_3p(alphas, *params_3p)
    drag_fn_used = drag_3p
    drag_params   = params_3p
    drag_label    = f"a + k*|sin(α)|^p  (p={p_fit:.4f})"
else:
    print(f"  → 2-param model sufficient (p≈1)")
    a_fit, k_fit = best_params_2p
    p_fit = 1.0
    drag_pred_alphas = drag_2p(alphas, *best_params_2p)
    drag_fn_used = drag_2p
    drag_params   = best_params_2p
    drag_label    = "a + k*|sin(α)|"

print(f"  --- Accuracy: Drag Factor ---")
drag_rmse, drag_r2, drag_acc = accuracy_report(noisy_drag, drag_pred_alphas, "Drag")

alpha_fine  = np.linspace(alphas.min(), alphas.max(), 10000)
drag_smooth = drag_fn_used(alpha_fine, *drag_params)
drag_interp = interp1d(alpha_fine, drag_smooth, kind='linear', fill_value='extrapolate')


# ============================================================
# 2. MAIN THRUST FITTING
# Physical constraint: thrust(0) = 0  →  only b*x^n is valid
# ============================================================

print("\n2. FITTING MAIN THRUST (ADVANCED)")
print("-"*70)

def model_power(x, b, n):
    """b * x^n  — satisfies thrust(0)=0 physically"""
    out = np.zeros_like(x, dtype=float)
    m = x > 0
    out[m] = b * np.power(x[m], n)
    return out

# Data prep
mask_main = (valves > 0.01) & (noisy_thrust_main > 0)
x_main    = valves[mask_main]
y_main    = noisy_thrust_main[mask_main]

# Log-log OLS — initial seed
log_x  = np.log(x_main)
log_y  = np.log(y_main)
c0     = np.polyfit(log_x, log_y, 1)
n_seed = c0[0]
b_seed = np.exp(c0[1])
print(f"  Log-log seed  : b={b_seed:.4f}, n={n_seed:.4f}")

# Global search with differential evolution (no local minima)
de_res = differential_evolution(
    lambda p: np.sum((y_main - model_power(x_main, *p))**2),
    bounds=[(1, 3000), (0.5, 6.0)],
    seed=42, maxiter=5000, tol=1e-14,
    mutation=(0.5, 1.5), recombination=0.9,
    popsize=20, polish=True
)
b_main, n_main = de_res.x
print(f"  DE global fit : b={b_main:.4f}, n={n_main:.4f}")

# Final polish with curve_fit starting from DE result
try:
    p_polish, _ = curve_fit(model_power, x_main, y_main,
                            p0=[b_main, n_main],
                            bounds=([0, 0.1], [5000, 7.0]),
                            maxfev=100000)
    # Only accept polish if it actually improves
    err_de     = np.sqrt(np.mean((y_main - model_power(x_main, b_main, n_main))**2))
    err_polish = np.sqrt(np.mean((y_main - model_power(x_main, *p_polish))**2))
    if err_polish < err_de:
        b_main, n_main = p_polish
        print(f"  Polished fit  : b={b_main:.4f}, n={n_main:.4f}")
except Exception:
    pass

print(f"\n  Final: b = {b_main:.8f}, n = {n_main:.8f}")
print(f"  --- Accuracy: Main Thrust ---")
main_pred = model_power(x_main, b_main, n_main)
main_rmse, main_r2, main_acc = accuracy_report(y_main, main_pred, "Main Thrust")
print(f"  Note: residual RMSE ~{main_rmse:.1f}N reflects data noise floor, not fit quality")

valve_fine         = np.linspace(0, valves.max(), 10000)
thrust_main_smooth = model_power(valve_fine, b_main, n_main)
thrust_main_interp = interp1d(valve_fine, thrust_main_smooth,
                               kind='linear', fill_value='extrapolate')


# ============================================================
# 3. ROTATIONAL THRUST FITTING
# ============================================================

print("\n3. FITTING ROTATIONAL THRUST (ADVANCED)")
print("-"*70)

mask_rot = (valves > 0.01) & (noisy_thrust_rot > 0)
x_rot    = valves[mask_rot]
y_rot    = noisy_thrust_rot[mask_rot]

log_x_rot  = np.log(x_rot)
log_y_rot  = np.log(y_rot)
coeffs_rot = np.polyfit(log_x_rot, log_y_rot, 1)
n_rot = coeffs_rot[0]
b_rot = np.exp(coeffs_rot[1])
print(f"  Log-log seed: b = {b_rot:.6f}, n = {n_rot:.6f}")

try:
    popt_rot, _ = curve_fit(model_power, x_rot, y_rot,
                            p0=[b_rot, n_rot],
                            bounds=([0, 0.1], [500, 5.0]),
                            maxfev=50000)
    b_rot, n_rot = popt_rot
except Exception:
    pass

print(f"  Refined fit : b = {b_rot:.6f}, n = {n_rot:.6f}")
print(f"  --- Accuracy: Rotational Thrust ---")
rot_rmse, rot_r2, rot_acc = accuracy_report(y_rot, model_power(x_rot, b_rot, n_rot), "Rot Thrust")

thrust_rot_smooth = model_power(valve_fine, b_rot, n_rot)
thrust_rot_interp = interp1d(valve_fine, thrust_rot_smooth,
                              kind='linear', fill_value='extrapolate')


# ============================================================
# TEST SOLUTIONS
# ============================================================

print("\n" + "="*70)
print("TESTING FITTED FUNCTIONS")
print("="*70)
test_solutions(drag_interp, thrust_main_interp, thrust_rot_interp)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "="*70)
print("FINAL PARAMETERS & ACCURACY SUMMARY")
print("="*70)
print(f"  Drag:      f(\u03b1) = {a_fit:.8f} + {k_fit:.8f} * |sin(\u03b1)|")
print(f"             R\u00b2 = {drag_r2:.6f}  |  RMSE = {drag_rmse:.6f}  |  Accuracy (R\u00b2) = {drag_acc:.4f}%")
print(f"\n  Main:      f(x) = {b_main:.8f} * x^{n_main:.8f}")
print(f"             R\u00b2 = {main_r2:.6f}  |  RMSE = {main_rmse:.6f}  |  Accuracy (R\u00b2) = {main_acc:.4f}%")
print(f"             RMSE ~{main_rmse:.1f}N on ~{np.mean(noisy_thrust_main[noisy_thrust_main>0]):.0f}N mean = {main_rmse/np.mean(noisy_thrust_main[noisy_thrust_main>0])*100:.2f}% relative error")
print(f"\n  Rotation:  f(x) = {b_rot:.8f} * x^{n_rot:.8f}")
print(f"             R\u00b2 = {rot_r2:.6f}  |  RMSE = {rot_rmse:.6f}  |  Accuracy (R\u00b2) = {rot_acc:.4f}%")
print("="*70)


# ============================================================
# VISUALIZATION
# ============================================================

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Sub-Problem 1: Fitted Functions & Residuals", fontsize=14)

# ---- Drag ----
axes[0, 0].scatter(alphas, noisy_drag, s=5, alpha=0.3, color='red', label='Noisy data')
axes[0, 0].plot(alpha_fine, drag_smooth, 'b-', linewidth=2, label=f'Fit  R²={drag_r2:.4f}')
axes[0, 0].set_xlabel('Angle α (rad)'); axes[0, 0].set_ylabel('Drag Multiplier')
axes[0, 0].set_title('Drag Factor'); axes[0, 0].legend(); axes[0, 0].grid(True)

axes[1, 0].scatter(alphas, noisy_drag - drag_fn_used(alphas, *drag_params),
                   s=5, alpha=0.5, color='green')
axes[1, 0].axhline(0, color='black', linestyle='--')
axes[1, 0].set_title(f'Drag Residuals  RMSE={drag_rmse:.4f}')
axes[1, 0].set_xlabel('Angle α (rad)'); axes[1, 0].set_ylabel('Residuals')
axes[1, 0].grid(True)

# ---- Main Thrust ----
axes[0, 1].scatter(valves, noisy_thrust_main, s=5, alpha=0.3, color='red', label='Noisy data')
axes[0, 1].plot(valve_fine, thrust_main_smooth, 'b-', linewidth=2,
                label=f'b·xⁿ  R²={main_r2:.4f}')
axes[0, 1].set_xlabel('Valve Position'); axes[0, 1].set_ylabel('Main Thrust (N)')
axes[0, 1].set_title('Main Thrust'); axes[0, 1].legend(); axes[0, 1].grid(True)

axes[1, 1].scatter(valves, noisy_thrust_main - model_power(valves, b_main, n_main),
                   s=5, alpha=0.5, color='green')
axes[1, 1].axhline(0, color='black', linestyle='--')
axes[1, 1].set_title(f'Main Thrust Residuals  RMSE={main_rmse:.4f}')
axes[1, 1].set_xlabel('Valve Position'); axes[1, 1].set_ylabel('Residuals')
axes[1, 1].grid(True)

# ---- Rotational Thrust ----
axes[0, 2].scatter(valves, noisy_thrust_rot, s=5, alpha=0.3, color='red', label='Noisy data')
axes[0, 2].plot(valve_fine, thrust_rot_smooth, 'b-', linewidth=2,
                label=f'b·xⁿ  R²={rot_r2:.4f}')
axes[0, 2].set_xlabel('Valve Position'); axes[0, 2].set_ylabel('Rotational Thrust (N)')
axes[0, 2].set_title('Rotational Thrust'); axes[0, 2].legend(); axes[0, 2].grid(True)

axes[1, 2].scatter(valves, noisy_thrust_rot - model_power(valves, b_rot, n_rot),
                   s=5, alpha=0.5, color='green')
axes[1, 2].axhline(0, color='black', linestyle='--')
axes[1, 2].set_title(f'Rotational Residuals  RMSE={rot_rmse:.4f}')
axes[1, 2].set_xlabel('Valve Position'); axes[1, 2].set_ylabel('Residuals')
axes[1, 2].grid(True)

plt.tight_layout()
plt.show()

print("\n✓ Complete!")
