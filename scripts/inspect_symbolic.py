"""Week 2, Task 1: inspect safe-control-gym's symbolic model interface."""
import casadi as cs

from safe_rl_cbf.envs import SafeCartpoleEnv

env = SafeCartpoleEnv()
model = env._env.symbolic

print("nx:", model.nx, " nu:", model.nu)
print("x_sym:", model.x_sym)
print("u_sym:", model.u_sym)
print("x_dot:", model.x_dot)

# Confirm the system is control-affine: d(x_dot)/du should not itself depend on u
dfdu = cs.jacobian(model.x_dot, model.u_sym)
print("control-affine:", not cs.depends_on(dfdu, model.u_sym))

# Confirm the state constraint bounds match our YAML
print("state upper bounds:", env._env.constraints.state_constraints[0].upper_bounds)
print("state lower bounds:", env._env.constraints.state_constraints[0].lower_bounds)