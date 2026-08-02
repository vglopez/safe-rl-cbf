"""Week 3: collect theta trajectories for baseline and filtered policies."""
import numpy as np
from stable_baselines3 import PPO

from safe_rl_cbf.envs import SafeCartpoleEnv
from safe_rl_cbf.filters.cbf_filter import AngleHOCBFFilter

N_EPISODES = 20
CONFIG = "configs/cartpole_cbf_eval.yaml"
THETA_MAX = 0.16
GAMMA = 20.0
OUT = "results/trajectories.npz"

def collect(model, filtered):
    env = SafeCartpoleEnv(config_path=CONFIG)
    raw_env = env._env
    cbf = None
    if filtered:
        cbf = AngleHOCBFFilter(
            raw_env.symbolic, theta_max=THETA_MAX, gamma1=GAMMA, gamma2=GAMMA,
            u_min=raw_env.physical_action_bounds[0][0],
            u_max=raw_env.physical_action_bounds[1][0],
        )
    episodes = []
    for _ in range(N_EPISODES):
        obs, info = env.reset()
        terminated = truncated = False
        thetas = [float(obs[2])]
        while not (terminated or truncated):
            u_norm, _ = model.predict(obs, deterministic=True)
            if cbf is not None:
                u_phys = raw_env.denormalize_action(u_norm)
                u_norm = raw_env.normalize_action(cbf.filter(obs, u_phys))
            obs, reward, terminated, truncated, info = env.step(u_norm)
            thetas.append(float(obs[2]))
        episodes.append(np.array(thetas))
    env.close()
    # Pad to equal length with NaN so we can store one 2-D array
    max_len = max(len(e) for e in episodes)
    padded = np.full((len(episodes), max_len), np.nan)
    for i, e in enumerate(episodes):
        padded[i, :len(e)] = e
    return padded

def main():
    model = PPO.load("models/ppo_baseline")
    baseline = collect(model, filtered=False)
    hocbf = collect(model, filtered=True)
    np.savez(OUT, baseline=baseline, hocbf=hocbf, theta_max=THETA_MAX)
    print(f"saved {OUT}: baseline {baseline.shape}, hocbf {hocbf.shape}")

if __name__ == "__main__":
    main()