"""Week 2 final evaluation: baseline vs. HOCBF filter (gamma=20), with
per-dimension violation breakdown."""
import numpy as np
from stable_baselines3 import PPO
import wandb

from safe_rl_cbf.envs import SafeCartpoleEnv
from safe_rl_cbf.filters.cbf_filter import AngleHOCBFFilter

N_EPISODES = 50
CONFIG = "configs/cartpole_cbf_eval.yaml"
THETA_MAX = 0.16
GAMMA = 20.0
BOUNDS = np.array([2.0, 2.0, 0.16, 1.0])
NAMES = ["x", "x_dot", "theta", "theta_dot"]

def run(model, filtered):
    env = SafeCartpoleEnv(config_path=CONFIG)
    raw_env = env._env
    cbf = None
    if filtered:
        cbf = AngleHOCBFFilter(
            raw_env.symbolic, theta_max=THETA_MAX, gamma1=GAMMA, gamma2=GAMMA,
            u_min=raw_env.physical_action_bounds[0][0],
            u_max=raw_env.physical_action_bounds[1][0],
        )
    per_dim = np.zeros(4, dtype=int)
    any_viol, total_return, interventions, failures, steps_total = 0, 0.0, 0, 0, 0

    for _ in range(N_EPISODES):
        obs, info = env.reset()
        terminated = truncated = False
        ret, steps = 0.0, 0
        while not (terminated or truncated):
            u_norm, _ = model.predict(obs, deterministic=True)
            if cbf is not None:
                u_phys = raw_env.denormalize_action(u_norm)
                u_safe_phys = cbf.filter(obs, u_phys)
                if abs(np.asarray(u_safe_phys).item() - np.asarray(u_phys).item()) > 1e-4:
                    interventions += 1
                u_norm = raw_env.normalize_action(u_safe_phys)
            obs, reward, terminated, truncated, info = env.step(u_norm)
            exceeded = np.abs(obs) > BOUNDS
            per_dim += exceeded.astype(int)
            any_viol += int(exceeded.any())
            ret += reward
            steps += 1
        steps_total += steps
        if steps < 150:
            failures += 1
        total_return += ret
    env.close()
    return per_dim, any_viol, total_return / N_EPISODES, interventions, failures, steps_total

def main():
    wandb.init(project="safe-rl-cbf", name="final-eval-week2", tags=["week2", "final"])
    model = PPO.load("models/ppo_baseline")

    results = {}
    for label, filtered in [("baseline", False), ("hocbf", True)]:
        per_dim, any_viol, avg_ret, interventions, failures, steps = run(model, filtered)
        results[label] = dict(per_dim=per_dim, any_viol=any_viol, avg_ret=avg_ret,
                              interventions=interventions, failures=failures, steps=steps)
        print(f"\n=== {label} ===")
        print(f"  total steps      : {steps}")
        print(f"  any-state viol   : {any_viol}  ({any_viol/N_EPISODES:.2f}/ep)")
        for name, count in zip(NAMES, per_dim):
            print(f"    {name:>10}    : {count}")
        print(f"  avg return       : {avg_ret:.2f}")
        print(f"  interventions    : {interventions}")
        print(f"  failures         : {failures}")

    b, h = results["baseline"], results["hocbf"]
    theta_red = 100 * (1 - h["per_dim"][2] / max(b["per_dim"][2], 1))
    any_red = 100 * (1 - h["any_viol"] / max(b["any_viol"], 1))
    print(f"\ntheta violations : {b['per_dim'][2]} -> {h['per_dim'][2]}  ({theta_red:.1f}% reduction)")
    print(f"any-state viol   : {b['any_viol']} -> {h['any_viol']}  ({any_red:.1f}% reduction)")
    print(f"return           : {b['avg_ret']:.2f} -> {h['avg_ret']:.2f}")

    wandb.log({"theta_reduction_pct": theta_red, "any_reduction_pct": any_red,
               "baseline_theta": int(b["per_dim"][2]), "hocbf_theta": int(h["per_dim"][2]),
               "baseline_return": b["avg_ret"], "hocbf_return": h["avg_ret"]})
    wandb.finish()

if __name__ == "__main__":
    main()