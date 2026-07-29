"""Week 2, Task 3 (fixed): evaluate PPO + HOCBF filter in physical action units."""
import numpy as np
from stable_baselines3 import PPO
import wandb

from safe_rl_cbf.envs import SafeCartpoleEnv
from safe_rl_cbf.filters.cbf_filter import AngleHOCBFFilter

N_EPISODES = 20
THETA_MAX = 0.16

def main():
    wandb.init(project="safe-rl-cbf", name="eval-ppo-filtered-v2",
               tags=["week2", "eval", "fixed-units"])

    env = SafeCartpoleEnv()
    raw_env = env._env
    model = PPO.load("models/ppo_baseline")
    cbf_filter = AngleHOCBFFilter(
        raw_env.symbolic, theta_max=THETA_MAX,
        u_min=raw_env.physical_action_bounds[0][0],
        u_max=raw_env.physical_action_bounds[1][0],
    )

    for ep in range(N_EPISODES):
        obs, info = env.reset()
        terminated = truncated = False
        ret, violations, steps, interventions = 0.0, 0, 0, 0
        while not (terminated or truncated):
            u_nom_norm, _ = model.predict(obs, deterministic=True)
            u_nom_phys = raw_env.denormalize_action(u_nom_norm)
            u_safe_phys = cbf_filter.filter(obs, u_nom_phys)
            u_safe_norm = raw_env.normalize_action(u_safe_phys)

            if np.abs(np.asarray(u_safe_phys).item() -
                      np.asarray(u_nom_phys).item()) > 1e-4:
                interventions += 1

            obs, reward, terminated, truncated, info = env.step(u_safe_norm)
            ret += reward
            violations += int(bool(info.get("constraint_violation", 0)))
            steps += 1
        wandb.log({"episode": ep, "return": ret, "violations": violations,
                   "steps": steps, "interventions": interventions})
        print(f"ep {ep}: return={ret:.2f} violations={violations} "
              f"steps={steps} interventions={interventions}")
    env.close()
    wandb.finish()

if __name__ == "__main__":
    main()