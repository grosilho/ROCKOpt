import jax.numpy as jnp
from solvers.opt.stabilized_utils.rho_estimator import rho_estimator
from solvers.opt.stabilized_utils.es_methods import RKW1, RKC1, RKU1
from utils.HelperClass import HelperClass


class SGF(HelperClass):
    """
    Stabilized Gradient Flow (SGF) method. This method solves the gradient flow ODE using a stabilized method.
    """

    def __init__(self, problem, params):

        default_params = {
            "delta": 0.1,
            "rho_freq": 5,
            "method": "RKC1",
            "damping": 0.05,
            "safe_add": 1,
            "log_history": False,
            "record_stages": False,
            "record_rejected": False,
        }
        super().__init__(default_params, params)

        self.description = "SGF"
        self.problem = problem

        method_mapping = {
            "RKC1": RKC1,
            "RKW1": RKW1,
            "RKU1": RKU1,
        }

        assert self.method in method_mapping, f"Stabilized method {self.method} not implemented."

        self.es = method_mapping[self.method](self.damping, self.safe_add)

        self.stats_keys = [
            "f_evals",
            "df_evals",
            "ddf_evals",
            "ddf_solves",
            "ddf_mults",
        ]
        self.history_keys = ["delta"]

    def pre_loop(self, x, fx):
        self.init_stats(self.stats_keys)
        self.init_history(self.history_keys)
        self.last_rho_eval_counter = 0
        self.rho_old = 0.0
        self.s_old = 0
        self.rho_estimator = rho_estimator(x.size)

        if self.log_history:
            self.append_to_history(delta=self.delta)

    def step(self, x, fx):
        """
        Solve the gradient flow ODE
        x'= -F.df(x)
        using the stabilized method.
        """

        problem = self.problem
        delta = self.delta

        def f(y):
            self.stats["df_evals"] += 1
            return -problem.df(y)

        self.update_rho_and_stages(x, delta)
        x, dx_norm = self.stabilized_iteration(f, x, delta)
        accepted = True

        if self.log_history and (accepted or self.record_rejected):
            self.append_to_history(delta=delta)

        fx_prev = fx
        fx = problem.f(x)
        self.stats["f_evals"] += 1

        f_diff = fx_prev - fx  # this should be positive
        x_norm = jnp.linalg.norm(x)
        fx_norm = jnp.abs(fx)

        convergence_info = {"f_diff": f_diff, "dx_norm": dx_norm, "x_norm": x_norm, "fx_norm": fx_norm}
        stepper_log_str = (
            f"eigval = {self.rho:.3e}, Δ = {delta:.3e}, s = {self.s}, dx = {dx_norm:.3e}, df = {f_diff:.3e}"
        )

        return x, fx, accepted, convergence_info, stepper_log_str

    def update_rho_and_stages(self, x, delta):
        # re-estimate rho every rho_freq iterations
        if self.last_rho_eval_counter % self.rho_freq == 0:
            # rho, n_f_eval = self.rho_estimator.rho(f, x, fx)
            self.rho, n_f_eval = self.rho_estimator.rho_linear_power_method(lambda v: self.problem.ddfv(x, v))

        # update coefficients if rho has changed
        if self.rho != self.rho_old:
            self.rho_old = self.rho
            self.s = self.es.get_s(delta * self.rho)
            if self.s != self.s_old:
                self.s_old = self.s
                self.es.update_coefficients(self.s)

        self.last_rho_eval_counter += 1

    def stabilized_iteration(self, f, x, delta):
        djm2, djm1 = 0.0, 0.0
        dj = self.es.mu[0] * delta * f(x)
        for j in range(2, self.s + 1):
            djm2, djm1, dj = djm1, dj, djm2
            dj = self.es.nu[j - 1] * djm1 + self.es.kappa[j - 1] * djm2 + self.es.mu[j - 1] * delta * f(x + djm1)

        x += dj
        dx_norm = jnp.linalg.norm(dj)

        return x, dx_norm
