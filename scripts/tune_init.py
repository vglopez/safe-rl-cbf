"""Find an init distribution: inside the safe set at reset, but hard enough
that unfiltered PPO still violates."""
import copy
import numpy as np
import yaml
from safe_control_gym.utils.registration import make
from stable_baselines3 import PPO

N_EPISODES = 20
BOUNDS = np.array([2.0, 2.0, 0.16, 1.0])
BASE = "configs/cartpole_cbf_eval.yaml"

# (x, x_dot, theta, theta_dot) half-widths
CANDIDATES = [
    (0.5, 1.5, 0.14, 0.95),   # best so far, for reference
    (0.5, 1.5, 0.145, 0.98),
    (0.5, 1.8, 0.15, 0.98),
    (0.8, 1.8, 0.15, 0.99),
    (0.8, 1.8, 0.155, 0.99),
]

def main():
    with open(BASE) as f:
        base_cfg = yaml.safe_load(f)["task_config"]
    model = PPO.load("models/ppo_baseline")

    print(f"{'theta_hw':>9} {'tdot_hw':>8} {'viol/ep':>8} {'theta_viol':>11} "
          f"{'failures':>9} {'oob_reset':>10}")
    for x_hw, xd_hw, th_hw, td_hw in CANDIDATES:
        cfg = copy.deepcopy(base_cfg)
        for key, hw in zip(["init_x", "init_x_dot", "init_theta", "init_theta_dot"],
                           [x_hw, xd_hw, th_hw, td_hw]):
            cfg["init_state_randomization_info"][key] = {
                "distrib": "uniform", "low": -hw, "high": hw}

        env = make("cartpole", **cfg)
        viol, theta_viol, failures, oob = 0, 0, 0, 0
        for _ in range(N_EPISODES):
            obs, info = env.reset()
            if np.any(np.abs(obs) > BOUNDS):
                oob += 1
            done, steps = False, 0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                exceeded = np.abs(obs) > BOUNDS
                viol += int(exceeded.any())
                theta_viol += int(exceeded[2])
                steps += 1
            if steps < 150:
                failures += 1
        env.close()
        print(f"{th_hw:9.2f} {td_hw:8.2f} {viol / N_EPISODES:8.2f} {theta_viol:11d} "
              f"{failures:9d} {oob:10d}")

if __name__ == "__main__":
    main()