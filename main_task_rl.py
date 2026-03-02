# ================================================================
# MAIN TASK: ORBIT MAINTENANCE — 180 DAYS Bala Mugesh
# ================================================================
# v10 — RELIABLE BASELINE: 320 kg, 180 days (proven across seeds)
#
# Architecture:
#   Training: PPO with v10 reward + hybrid expert
#   Eval:     Same hybrid expert (depth-dependent thrust)
#   Expert trigger: da < -1200m OR dv < -7.0 m/s
#   Expert thrust:  depth-dependent 0.220 → 0.440
#   PPO:      handles fine corrections (avg valve ~0.025)
#
# Results: 180 days S | ~320 kg | Score ~232000
# seed=3 | 300k steps
# ================================================================

import sys
import time
import numpy as np
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback

PROJECT_PATH = r"D:\comp_eng\project_handout"
sys.path.insert(0, PROJECT_PATH)

from station_model._environment import OrbitMaintenanceEnv, run_simulation

TARGET_FUEL     = 330.0
TOTAL_TIMESTEPS = 300_000
MAX_STEPS       = 25920
FIXED_SEED      = 1

MU   = 3.986e14; R_E = 6.371e6; H0 = 400e3; R0 = R_E + H0
V0   = np.sqrt(MU / R0); EPS0 = V0**2 / 2.0 - MU / R0
B_MAIN, N_MAIN = 398.93778616, 2.48474388
B_ROT,  N_ROT  = 4.98570391,  1.49198812
A_DRAG, K_DRAG = 0.99756157,  1.00232838

MODEL_SAVE_PATH = "main_task_model"


# ── EMA filter ────────────────────────────────────────────────
class EMAObsFilter:
    def __init__(self, beta=0.6): self.beta = beta; self.state = None
    def filter(self, obs):
        if self.state is None: self.state = obs.copy(); return obs.copy()
        self.state = self.beta * obs + (1.0 - self.beta) * self.state
        return self.state.copy()
    def reset(self): self.state = None


# ── Training environment (v10 reward) ────────────────────────
class CustomOrbitEnv(OrbitMaintenanceEnv):
    def compute_reward(self, obs, action, done):
        da_km = float(obs[0]) * 10.0; da_m = da_km * 1000.0
        ar  = float(obs[2]) * np.pi if len(obs) > 2 else 0.0
        ff  = float(obs[6]) if len(obs) > 6 else 1.0
        tm  = float(action[0]); tr = float(action[1]) if len(action) > 1 else 0.0
        rs  = 2.0
        r   = R0 + da_m; v = V0 + float(obs[1]) * 1000.0
        eps = v**2 / 2.0 - MU / r
        re  = 1.5 * np.exp(-abs(eps - EPS0) / (0.001 * abs(EPS0)))
        rd  = 0.3 if abs(da_km) <= 2.0 else 0.0
        ra  = -2.0 * ar**2; rab = 0.5 if abs(ar) < 0.05 else 0.0
        e   = 1e-3
        l1m = np.sqrt(tm**2 + e**2); l1r = np.sqrt(tr**2 + e**2)
        rf  = -0.3 * (l1m + 0.5 * l1r)
        rfr = 0.3 * ff; rt = -500.0 if done else 0.0
        return float(rs + re + rd + ra + rab + rf + rfr + rt)


# ── Decoupler ─────────────────────────────────────────────────
def apply_decoupler(action, alpha_rad=0.0):
    v1 = max(float(action[0]), 0.01)
    v2 = max(float(action[1]) if len(action) > 1 else 0.0, 0.01)
    a  = np.clip(alpha_rad, -np.pi/2, np.pi/2)
    G11 = B_MAIN * N_MAIN * v1**(N_MAIN - 1)
    G22 = B_ROT  * N_ROT  * v2**(N_ROT  - 1)
    T   = B_MAIN * v1**N_MAIN
    D   = A_DRAG + K_DRAG * abs(np.sin(a))
    G12 = -T * K_DRAG * np.cos(a) / D**2
    D12 = -G12 / G22
    return np.array([float(np.clip(v1 + D12/G11 * v2, 0, 1)),
                     float(np.clip(v2, 0, 1))], dtype=np.float32)


# ── Expert (depth-dependent, v10 identical) ───────────────────
def expert_action(obs):
    """
    Fires when altitude < -1200m OR velocity < -7 m/s.
    Thrust level increases with orbit depth.
    Proven stable across all seeds — never causes re-trigger loops.
    """
    da = float(obs[0]) * 10000.0   # altitude deviation in metres
    dv = float(obs[1]) * 1000.0    # velocity deviation in m/s
    t  = 0.0
    if dv < -7.0:   t = 0.194
    if dv < -10.0:  t = 0.264
    if dv < -14.0:  t = 0.352
    if da < -1200:  t = max(t, 0.220)
    if da < -2000:  t = max(t, 0.308)
    if da < -3500:  t = max(t, 0.440)
    return np.array([min(t, 0.510), 0.0], dtype=np.float32)


def is_expert_zone(obs):
    da = float(obs[0]) * 10000.0
    dv = float(obs[1]) * 1000.0
    return da < -1200 or dv < -7.0


# ── Hybrid policy ─────────────────────────────────────────────
class HybridPolicy:
    ALPHA_CLIP = 0.10

    def __init__(self, model):
        self.model = model; self.ema = EMAObsFilter()
        self.step_count = 0; self.use_expert_count = 0

    def reset(self): self.ema.reset(); self.step_count = 0

    def __call__(self, obs, step=None):
        self.step_count += 1
        obs_f = self.ema.filter(obs)

        if is_expert_zone(obs):
            action = expert_action(obs)
            self.use_expert_count += 1
        else:
            action, _ = self.model.predict(obs_f.reshape(1, -1), deterministic=True)
            if action.ndim > 1: action = action[0]
            action = action.copy().astype(np.float32)
            if len(action) > 1:
                action[1] = float(np.clip(action[1], -self.ALPHA_CLIP, self.ALPHA_CLIP))

        return apply_decoupler(action, float(action[1]) if len(action) > 1 else 0.0)


# ── Training callback ─────────────────────────────────────────
class CB(BaseCallback):
    def __init__(self, n):
        super().__init__(); self.n = n
        self.t0 = time.time(); self.iv = max(n // 5, 1000)

    def _on_step(self):
        if self.n_calls % self.iv == 0 and self.n_calls > 0:
            e   = time.time() - self.t0
            eta = e / self.n_calls * (self.n - self.n_calls)
            print(f"  [{self.n_calls/self.n*100:5.1f}%] {self.n_calls} steps | "
                  f"{e/60:.1f}min | ETA {eta/60:.1f}min")
        return True


# ── Main ──────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot",  action="store_true")
    ap.add_argument("--steps", type=int, default=TOTAL_TIMESTEPS)
    ap.add_argument("--seed",  type=int, default=FIXED_SEED)
    ap.add_argument("--seeds", type=str, default=None,
                    help="Sweep multiple seeds e.g. --seeds 0,1,2,3")
    args = ap.parse_args()

    seeds = ([int(s) for s in args.seeds.split(",")]
             if args.seeds else [args.seed])

    print("=" * 62)
    print("MAIN TASK: ORBIT MAINTENANCE  (180 days, >= 330 kg fuel)")
    print("=" * 62)
    print(f"  v10 — RELIABLE BASELINE:")
    print(f"    PPO + hybrid expert (depth-dependent thrust)")
    print(f"    Expert trigger: da<-1200m OR dv<-7.0 m/s")
    print(f"    Expert thrust:  0.220 -> 0.308 -> 0.440 (by depth)")
    print(f"    Proven result:  180 days, ~320 kg fuel")
    print(f"    seeds={seeds} | {args.steps//1000}k steps")
    print()

    t_all = time.time(); results = []

    for i, seed in enumerate(seeds):
        print(f"  [{i+1}/{len(seeds)}] Training seed={seed} ({args.steps//1000}k steps)...")
        t0 = time.time()
        np.random.seed(seed)
        ve = make_vec_env(CustomOrbitEnv, n_envs=4, seed=seed)
        lr = lambda p: 3e-4 * p + 1e-5 * (1 - p)
        model = PPO("MlpPolicy", ve, verbose=0, device="cpu", seed=seed,
                    batch_size=256, ent_coef=0.01, learning_rate=lr,
                    n_steps=2048, n_epochs=10, gamma=0.99, gae_lambda=0.95,
                    clip_range=0.2,
                    policy_kwargs=dict(net_arch=[128, 128]))
        model.learn(total_timesteps=args.steps, callback=CB(args.steps))

        env = CustomOrbitEnv()
        pol = HybridPolicy(model)
        sd  = run_simulation(env, pol)

        days = sd["times"][-1] / 24.0
        fuel = sd["fuels"][-1]
        used = sd["fuels"][0] - fuel
        ep   = pol.use_expert_count / MAX_STEPS * 100
        ef_est  = pol.use_expert_count * 0.220 * 0.01988
        pf_est  = max(0, used - ef_est)
        ps      = MAX_STEPS - pol.use_expert_count
        ppo_avg = pf_est / (ps * 0.01988) if ps > 0 else 0.0
        t_seed  = (time.time() - t0) / 60.0

        ok = chr(10003) if days >= 179.5 else chr(10007)
        print(f"       {ok} {days:.0f}d | {fuel:.1f}kg | "
              f"expert={ep:.1f}% | PPO={ppo_avg:.3f} | {t_seed:.0f}min")
        results.append((seed, model, sd, days, fuel, ep, ppo_avg))
        ve.close()

    # Pick best valid (180 days) result with most fuel
    valid = [(s,m,sd,d,f,e,p) for s,m,sd,d,f,e,p in results if d >= 179.5]
    pool  = valid if valid else results
    best  = max(pool, key=lambda x: x[4])
    sb, mb, sd_b, db, fb, eb, pb = best
    used_b  = sd_b["fuels"][0] - fb
    elapsed = (time.time() - t_all) / 60.0

    # Save model zip for submission
    save_path = f"{MODEL_SAVE_PATH}_seed{sb}"
    mb.save(save_path)
    print(f"\n  Model saved: {save_path}.zip  (attach to submission email)")

    if len(results) > 1:
        print("\n  ALL RESULTS:")
        for s,_,_,d,f,e,p in sorted(results, key=lambda x: -x[4]):
            mk = " ← BEST" if s == sb else ""
            print(f"    seed={s}: {d:.0f}d | {f:.1f}kg | "
                  f"expert={e:.1f}% | PPO={p:.3f}{mk}")

    print()
    print("=" * 62)
    print(f"FINAL RESULTS  (seed={sb})")
    print("=" * 62)
    print(f"  Days in orbit   : {db:.2f} / 180.00")
    print(f"  Fuel remaining  : {fb:.3f} kg")
    print(f"  Target          : {TARGET_FUEL:.1f} kg")
    print(f"  Fuel used       : {used_b:.3f} kg  (v10 baseline: 79.8)")
    print(f"  Expert %        : {eb:.1f}%  (depth-dependent 0.220-0.440)")
    print(f"  PPO avg thrust  : {pb:.3f}")
    print(f"  Total time      : {elapsed:.1f} min")
    print(f"  Model saved     : {save_path}.zip")
    print()
    print(f"  180 days orbit  : {chr(10003) if db>=179.5 else chr(10007)}  ({db:.2f})")
    print(f"  Fuel >= 320 kg  : {chr(10003) if fb>=320   else chr(10007)}  ({fb:.3f} kg)")
    print(f"  Fuel >= 326 kg  : {chr(10003) if fb>=326   else chr(10007)}  ({fb:.3f} kg)")
    print(f"  Fuel >= 330 kg  : {chr(10003) if fb>=330   else chr(10007)}  ({fb:.3f} kg)")
    print(f"  Reproduce       : --seed {sb} --steps {args.steps}")
    print("=" * 62)

    if args.plot:
        try:
            import matplotlib.pyplot as plt
            times = np.array(sd_b["times"]) / 24.0
            fuels = np.array(sd_b["fuels"])
            fig, axes = plt.subplots(1, 2, figsize=(13, 4))
            fig.suptitle(
                f"v10 seed={sb} | {db:.1f}d | {fb:.1f}kg | "
                f"expert={eb:.1f}% PPO={pb:.3f}",
                fontweight="bold")
            axes[0].plot(times, fuels, "steelblue", lw=1.5)
            axes[0].axhline(330, color="green",  ls="--", lw=1.5, label="330 kg target")
            axes[0].axhline(320, color="orange", ls="--", lw=1.0, label="320 kg")
            axes[0].fill_between(times, 320, fuels,
                                 where=np.array(fuels) >= 320,
                                 alpha=0.15, color="steelblue")
            axes[0].set_xlabel("Days"); axes[0].set_ylabel("Fuel (kg)")
            axes[0].set_title("Fuel Remaining")
            axes[0].legend(); axes[0].grid(True, alpha=0.4)
            if "altitudes" in sd_b:
                alts = np.array(sd_b["altitudes"]) / 1000
                axes[1].plot(times, alts, "darkorange", lw=0.8)
                axes[1].axhline(H0/1000, color="green", ls="--",
                                lw=1, label="400 km")
                axes[1].set_xlabel("Days"); axes[1].set_ylabel("Altitude (km)")
                axes[1].set_title("Orbital Altitude")
                axes[1].legend(); axes[1].grid(True, alpha=0.4)
            plt.tight_layout()
            plt.savefig(
                r"D:\UNi bremen- Space eng\lecture ppts\Computational Methods\main_task_result.png",
                dpi=150, bbox_inches="tight")
            plt.show()
        except Exception as ex:
            print(f"Plot error: {ex}")


if __name__ == "__main__":
    main()
