"""Week 1: train an unconstrained PPO policy and log training progress."""
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from wandb.integration.sb3 import WandbCallback

import wandb
from safe_rl_cbf.envs import SafeCartpoleEnv

TOTAL_TIMESTEPS = 200_000

def main():
    run = wandb.init(project="safe-rl-cbf", name="ppo-baseline",
                      tags=["week1", "baseline"], sync_tensorboard=True)

    env = Monitor(SafeCartpoleEnv())
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=f"runs/{run.id}")
    model.learn(total_timesteps=TOTAL_TIMESTEPS,
                callback=WandbCallback(verbose=2))

    model.save("models/ppo_baseline")
    env.close()
    run.finish()

if __name__ == "__main__":
    main()