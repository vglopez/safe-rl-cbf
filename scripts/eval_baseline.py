"""Week 1: evaluate the trained PPO baseline against the safety constraint."""
from stable_baselines3 import PPO

import wandb
from safe_rl_cbf.envs import SafeCartpoleEnv

N_EPISODES = 20

def main():
    wandb.init(project="safe-rl-cbf", name="eval-ppo-baseline",
               tags=["week1", "eval"])

    env = SafeCartpoleEnv()
    model = PPO.load("models/ppo_baseline")

    for ep in range(N_EPISODES):
        obs, info = env.reset()
        terminated = truncated = False
        ret, violations, steps = 0.0, 0, 0
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ret += reward
            violations += int(bool(info.get("constraint_violation", 0)))
            steps += 1
        wandb.log({"episode": ep, "return": ret,
                   "violations": violations, "steps": steps})
        print(f"ep {ep}: return={ret:.2f} violations={violations} steps={steps}")
    env.close()
    wandb.finish()

if __name__ == "__main__":
    main()