"""Which state dimension is actually causing the counted violations?"""
import numpy as np
from stable_baselines3 import PPO

from safe_rl_cbf.envs import SafeCartpoleEnv

N_EPISODES = 20
BOUNDS = np.array([2.0, 2.0, 0.16, 1.0])
NAMES = ["x", "x_dot", "theta", "theta_dot"]

def main():
    env = SafeCartpoleEnv(config_path="configs/cartpole_cbf_eval.yaml")
    model = PPO.load("models/ppo_baseline")
    per_dim = np.zeros(4, dtype=int)
    any_dim, steps_total, at_reset = 0, 0, 0

    for _ in range(N_EPISODES):
        obs, info = env.reset()
        if np.any(np.abs(obs) > BOUNDS):
            at_reset += 1
        terminated = truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            exceeded = np.abs(obs) > BOUNDS
            per_dim += exceeded.astype(int)
            any_dim += int(exceeded.any())
            steps_total += 1
    env.close()

    print(f"total steps: {steps_total}, steps with any exceedance: {any_dim}")
    print(f"episodes already out of bounds at reset: {at_reset}/{N_EPISODES}")
    for name, count in zip(NAMES, per_dim):
        print(f"  {name:>10}: {count:5d} steps ({100*count/steps_total:5.1f}%)")

if __name__ == "__main__":
    main()