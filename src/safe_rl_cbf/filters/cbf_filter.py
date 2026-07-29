"""High-Order Control Barrier Function (HOCBF) QP safety filter for cart-pole
angle stabilization. See Xiao & Belta, 'Control Barrier Functions for Systems
with High Relative Degree' (2019) for the general construction.
"""
import casadi as cs
import numpy as np


class AngleHOCBFFilter:
    def __init__(self, symbolic_model, theta_max, gamma1=5.0, gamma2=5.0,
                 slack_weight=1e4, u_min=-1.0, u_max=1.0):
        X = symbolic_model.x_sym
        u = symbolic_model.u_sym
        x_dot = symbolic_model.x_dot
        theta = X[2]  # state = [x, x_dot, theta, theta_dot]

        # Two one-sided barriers. Gradient is constant (∓1), so control
        # authority never vanishes inside the safe set — unlike the symmetric
        # quadratic barrier, whose gradient dies at theta = 0.
        conditions, h_exprs = [], []
        for h in (theta_max - theta, theta_max + theta):
            h_dot = cs.dot(cs.gradient(h, X), x_dot)
            psi = h_dot + gamma1 * h
            psi_dot = cs.dot(cs.gradient(psi, X), x_dot)
            conditions.append(psi_dot + gamma2 * psi)
            h_exprs.append(h)

        self._hocbf_conditions = cs.Function(
            "hocbf_conditions", [X, u], [cs.vertcat(*conditions)],
            ["X", "u"], ["value"],
        )
        self._h_func = cs.Function("h", [X], [cs.vertcat(*h_exprs)], ["X"], ["h"])

        self.nx, self.nu = symbolic_model.nx, symbolic_model.nu
        self.u_min, self.u_max = u_min, u_max
        self.slack_weight = slack_weight
        self._build_qp()

    def _build_qp(self):
        opti = cs.Opti("conic")
        u_var = opti.variable(self.nu, 1)
        slack = opti.variable(2, 1)
        x_param = opti.parameter(self.nx, 1)
        u_nom_param = opti.parameter(self.nu, 1)

        conditions = self._hocbf_conditions(X=x_param, u=u_var)["value"]
        cost = 0.5 * cs.norm_2(u_var - u_nom_param) ** 2 + self.slack_weight * cs.sumsqr(slack)
        opti.subject_to(conditions + slack >= 0)
        opti.subject_to(slack >= 0)
        opti.subject_to(opti.bounded(self.u_min, u_var, self.u_max))
        opti.minimize(cost)
        opti.solver("qpoases", {"printLevel": "none", "error_on_fail": False})

        self._opti = opti
        self._u_var, self._slack = u_var, slack
        self._x_param, self._u_nom_param = x_param, u_nom_param

    def filter(self, state: np.ndarray, u_nominal: np.ndarray) -> np.ndarray:
        self._opti.set_value(self._x_param, state)
        self._opti.set_value(self._u_nom_param, u_nominal)
        try:
            sol = self._opti.solve()
            return np.array(sol.value(self._u_var)).reshape(self.nu)
        except RuntimeError:
            # Infeasible: fall back to the nominal action clipped to bounds
            # rather than letting the filter crash the episode.
            return np.clip(u_nominal, self.u_min, self.u_max)