"""Week 0 smoke test: random policy on constrained cart-pole, logged to W&B."""
import wandb
import yaml
from safe_control_gym.utils.registration import make

N_EPISODES = 5
CONFIG_PATH = "configs/cartpole_stab.yaml"

def main():
    with open(CONFIG_PATH, "r") as f:
        task_config = yaml.safe_load(f)["task_config"]

    wandb.init(project="safe-rl-cbf", name="smoke-random-policy",
               tags=["week0", "baseline"])

    env = make("cartpole", **task_config)
    for ep in range(N_EPISODES):
        _obs, info = env.reset()
        done, ret, violations, steps = False, 0.0, 0, 0
        while not done:
            action = env.action_space.sample()
            _obs, reward, done, info = env.step(action)
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