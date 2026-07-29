"""Check for a unit mismatch between normalized action space and physical dynamics."""
from safe_rl_cbf.envs import SafeCartpoleEnv

env = SafeCartpoleEnv()
raw_env = env._env

print("action_space:", raw_env.action_space)
print("physical_action_bounds:", getattr(raw_env, "physical_action_bounds", "NOT FOUND"))

# Look for any normalize/denormalize/scale helpers
candidates = [a for a in dir(raw_env) if "norm" in a.lower() or "scale" in a.lower()]
print("candidate methods/attrs:", candidates)