"""Week 2: fair, matched comparison of gain settings vs. the unfiltered baseline.
A fresh environment is constructed for each condition so all conditions see the
same sequence of (seeded) random initial states.
"""
from stable_baselines3 import PPO

from safe_rl_cbf.envs import SafeCartpoleEnv
from safe_rl_cbf.filters.cbf_filter import AngleHOCBFFilter

N_EPISODES = 20
CONDITIONS = [
    ("no_filter", None),
    ("gamma_10_10", (10.0, 10.0)),
    ("gamma_20_20", (20.0, 20.0)),
    ("gamma_40_40", (40.0, 40.0)),
    ("gamma_80_80", (80.0, 80.0)),
]

def run_condition(model, gains):
    # fresh construction -> same seeded reset sequence
    env = SafeCartpoleEnv(config_path="configs/cartpole_cbf_eval.yaml")
    raw_env = env._env
    cbf_filter = None
    if gains is not None:
        cbf_filter = AngleHOCBFFilter(
            raw_env.symbolic, theta_max=0.16, gamma1=gains[0], gamma2=gains[1],
            u_min=raw_env.physical_action_bounds[0][0],
            u_max=raw_env.physical_action_bounds[1][0],
        )
    total_violations, total_return, total_interventions, failures = 0, 0.0, 0, 0
    for _ in range(N_EPISODES):
        obs, info = env.reset()
        terminated = truncated = False
        ret, violations, steps = 0.0, 0, 0
        while not (terminated or truncated):
            u_nom_norm, _ = model.predict(obs, deterministic=True)
            if cbf_filter is not None:
                u_nom_phys = raw_env.denormalize_action(u_nom_norm)
                u_safe_phys = cbf_filter.filter(obs, u_nom_phys)
                u_safe_norm = raw_env.normalize_action(u_safe_phys)
                if abs(float(u_safe_phys) - float(u_nom_phys)) > 1e-4:
                    total_interventions += 1
            else:
                u_safe_norm = u_nom_norm
            obs, reward, terminated, truncated, info = env.step(u_safe_norm)
            ret += reward
            violations += int(bool(info.get("constraint_violation", 0)))
            steps += 1
        if steps < 150:
            failures += 1
        total_violations += violations
        total_return += ret
    env.close()
    return total_violations, total_return / N_EPISODES, total_interventions, failures

def main():
    model = PPO.load("models/ppo_baseline")
    print(f"{'condition':>12} {'violations':>11} {'viol/ep':>8} {'avg_return':>11} "
          f"{'interventions':>13} {'failures':>9}")
    for name, gains in CONDITIONS:
        violations, avg_return, interventions, failures = run_condition(model, gains)
        print(f"{name:>12} {violations:11d} {violations / N_EPISODES:8.2f} "
              f"{avg_return:11.2f} {interventions:13d} {failures:9d}")

if __name__ == "__main__":
    main()