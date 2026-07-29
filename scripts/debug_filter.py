"""Week 2 diagnostic: log theta and the filter's proposed correction along a
real (unfiltered) trajectory, to check the correction's direction makes sense.
"""
import numpy as np
from stable_baselines3 import PPO

from safe_rl_cbf.envs import SafeCartpoleEnv
from safe_rl_cbf.filters.cbf_filter import AngleHOCBFFilter

env = SafeCartpoleEnv()
raw_env = env._env
model = PPO.load("models/ppo_baseline")
cbf_filter = AngleHOCBFFilter(
    raw_env.symbolic, theta_max=0.16,
    u_min=raw_env.physical_action_bounds[0][0],
    u_max=raw_env.physical_action_bounds[1][0],
)

obs, info = env.reset()
terminated = truncated = False
step = 0
while not (terminated or truncated) and step < 60:
    theta, theta_dot = obs[2], obs[3]
    h_val = float(cbf_filter._h_func(X=obs)["h"])
    u_nom_norm, _ = model.predict(obs, deterministic=True)
    u_nom_phys = raw_env.denormalize_action(u_nom_norm)
    u_safe_phys = cbf_filter.filter(obs, u_nom_phys)
    print(f"step={step:3d} theta={theta:+.4f} theta_dot={theta_dot:+.4f} "
          f"h={h_val:+.4f} u_nom={float(np.asarray(u_nom_phys)):+.3f} "
          f"u_safe={float(np.asarray(u_safe_phys)):+.3f}")
    obs, reward, terminated, truncated, info = env.step(u_nom_norm)  # unfiltered step
    step += 1
env.close()