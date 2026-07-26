from safe_rl_cbf.envs import SafeCartpoleEnv

env = SafeCartpoleEnv()
obs, info = env.reset()
print("obs shape:", obs.shape, "action space:", env.action_space)
for _ in range(5):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    print(f"reward={reward:.3f} terminated={terminated} truncated={truncated}")
env.close()