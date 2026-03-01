## Sub-Problem 1: System Identification — Thrust & Drag Curve Fitting

### Problem
Fit mathematical models to noisy measurement data from three physical
components of the space station:
1. **Drag factor** as a function of attitude angle α
2. **Main thruster** thrust as a function of valve opening
3. **Rotational thruster** thrust as a function of valve opening

All data contains realistic sensor noise. Goal: find parameters that
best describe the underlying physics, not just memorise the noise.

---

### Step 1 — Drag Factor Fitting

**Model:** `f(α) = a + k · |sin(α)|`

Physical basis: drag increases with the projected cross-section area,
which is proportional to |sin(α)| at attitude angle α.

**Approach:**
- Started with basic `scipy.curve_fit` using multiple initial guesses
  to avoid local minima: p0 = [1.0, 0.5], [1.0, 1.0], [1.2, 0.8], [0.8, 1.2]
- Best result selected by lowest RMSE across all starting points
- Extended to 3-parameter model `a + k·|sin(α)|^p` using
  Differential Evolution (global search) + curve_fit polish to test
  whether p≠1 could improve accuracy
- Model selection: 3-param used only if RMSE strictly lower than 2-param

**Result:**
```
f(α) = 0.99756157 + 1.00232838 · |sin(α)|
R² = 0.9733  |  RMSE = 0.0515  |  Accuracy = 97.33%
```

**Why 97.33% is the noise floor:**
SNR = R²/(1-R²) = 36.5 (15.6 dB). Reaching 99% R² would require
8× less noise. Any higher fit = overfitting, not real improvement.
Polynomial/spline/neural approaches would memorise noise — wrong for physics.

---

### Step 2 — Main Thruster Fitting

**Model:** `f(x) = B · x^N` (power law)

Physical basis: rocket thrust follows a power law with valve opening.

**Approach:**
- **Stage 1 — Log-log seed:** `log(thrust) = log(B) + N·log(x)`
  Transformed to linear regression for a fast, reliable initial estimate.
  Gave: B=231.2, N=1.672 (good start, not final)
- **Stage 2 — Differential Evolution (global search):**
  Searched bounds B∈[100,600], N∈[1.0,4.0] with 3000 iterations
  to escape the log-log local minimum caused by noise near x=0.
  Gave: B=398.94, N=2.485
- **Stage 3 — Polish:** curve_fit from DE result for fine-tuning.
  Two-pass optimisation: coarse (ftol=1e-14) then fine (ftol=1e-16)

**Why log-log alone fails:** noise near valve=0 distorts the linear
regression in log-space. DE global search finds the true minimum.

**Result:**
```
f(x) = 398.93778616 · x^2.48474388
R² = 0.9988  |  RMSE = 4.01 N  |  Accuracy = 99.88%
RMSE ~4N on ~117N mean = 3.43% relative error (noise floor)
```

---

### Step 3 — Rotational Thruster Fitting

**Model:** `f(x) = B · x^N` (same power law form)

**Approach:**
- Same 3-stage pipeline as main thruster
- Log-log seed: B=4.606, N=1.404
- Differential Evolution: refined to B=4.986, N=1.492
- curve_fit polish from DE result

**Result:**
```
f(x) = 4.98570391 · x^1.49198812
R² = 0.9989  |  RMSE = 0.0498  |  Accuracy = 99.89%
```

---

### Accuracy Metric Used

Standard MAPE was unreliable because near-zero thrust values (small
valve openings) produce extremely large percentage errors even for
tiny absolute errors. Used **R²-based accuracy** instead:
```
Accuracy = R² × 100%
```

Also reported **Median APE** (MdAPE) using only values above 5% of
maximum — robust to low-value outliers.

---

### Final Results

| Component | Model | R² | Accuracy |
|-----------|-------|----|----------|
| Drag factor | a + k·\|sin(α)\| | 0.9733 | 97.33% |
| Main thrust | B·x^N | 0.9988 | 99.88% |
| Rotational thrust | B·x^N | 0.9989 | 99.89% |

All L2-norm tests passed against ground truth:
- Drag L2 error: 0.017032
- Main thrust L2 error: 4.214879
- Rotational thrust L2 error: 0.055201

### Key Insight
Log-log linearisation is a good starting point but fails for noisy
power-law data near zero. Differential Evolution global search followed
by gradient-based polish (two-pass L-BFGS-B) is the reliable solution
for identifying nonlinear physical models from noisy measurements.
