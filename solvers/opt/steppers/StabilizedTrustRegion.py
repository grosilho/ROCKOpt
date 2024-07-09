import jax.numpy as jnp
from solvers.opt.stabilized_utils.rho_estimator import rho_estimator
from solvers.opt.stabilized_utils.es_methods import RKW1, RKC1, RKU1
from solvers.opt.steppers.TrustRegion import TR


class STR(TR):
    def __init__(self, problem, params):

        default_params = {
            "delta_max": 1.0,
            "delta_init": 0.1,
            "eta": 1e-4,
            "log_history": False,
            "record_rejected": False,
            "method": "RKC1",
            "dt": 1.0,
            "damping": 1.0,
            "safe_add": 1,
            "rho_freq": 1,
            "max_steps": 100,
            "rel_res_tol": 1e-3,
            "abs_tol": 1e-9,
            "rel_tol": 1e-3,
        }
        super(TR, self).__init__(default_params, params)

        self.description = "STR"
        self.problem = problem

        method_mapping = {
            "RKC1": RKC1,
            "RKW1": RKW1,
            "RKU1": RKU1,
        }
        assert self.method in method_mapping, f"Stabilized method {self.method} not implemented."

        self.es = method_mapping[self.method](self.damping, self.safe_add)

        self.compute_direction = self.dog_leg

        self.stats_keys = [
            "f_evals",
            "df_evals",
            "ddf_evals",
            "ddf_solves",
            "ddf_mults",
        ]
        self.history_keys = ["delta"]

    def pre_loop(self, x, fx):
        super().pre_loop(x, fx)
        self.last_rho_eval_counter = 0
        self.rho_old = 0.0
        self.s_old = 0
        self.rho_estimator = rho_estimator(x.size)

    def newton_direction(self, p, x):
        """
        Do some steps of size dt on the equation
        p' = -g - B @ p , p(0) = p
        using the explicit stabilized method.
        The equilibrium of this equation is at p'==0, hence p = -B^-1 @ g, i.e. the newton step.
        Thus, doing a few steps on this equation we approximate the newton step.
        Compared to standard method:
        Advantage: no need to invert B.
        Disadvantages: need to estimate the spectral radius of B and compute some matrix-vector products.
        """

        problem = self.problem
        g = problem.df(x)
        self.stats["df_evals"] += 1

        def Bv(v):
            self.stats["ddf_mults"] += 1
            return problem.ddfv(x, v)

        def f(y):
            self.stats["ddf_mults"] += 1
            return -g - problem.ddfv(x, y)

        res0 = jnp.linalg.norm(f(p))
        x_norm = jnp.linalg.norm(x)

        self.update_rho_and_stages(Bv, self.dt)

        i = 0
        while True:
            p = self.stabilized_iteration(f, p, self.dt, self.s)
            i += 1

            res = jnp.linalg.norm(f(p))
            if res < res0 * self.rel_res_tol or res < self.abs_tol or res < x_norm * self.rel_tol:
                break
            if i >= self.max_steps:
                self.logger.warning("Stabilized Newton direction reached maximum iterations.")
                break

        self.logger.info(
            f"Stabilized Newton direction with s = {self.s} and eigval = {self.rho:.2f}, converged in {i} iterations."
        )

        return p, g, Bv

    def stabilized_iteration(self, f, p, dt, s):
        djm2, djm1 = 0.0, 0.0
        dj = self.es.mu[0] * dt * f(p)
        for j in range(2, s + 1):
            djm2, djm1, dj = djm1, dj, djm2
            dj = self.es.nu[j - 1] * djm1 + self.es.kappa[j - 1] * djm2 + self.es.mu[j - 1] * dt * f(p + djm1)
        p += dj
        return p

    def update_rho_and_stages(self, Bv, delta):
        # re-estimate rho every rho_freq iterations
        if self.last_rho_eval_counter % self.rho_freq == 0:
            # rho, n_f_eval = self.rho_estimator.rho(f, x, fx)
            self.rho, n_f_eval = self.rho_estimator.rho_linear_power_method(Bv)

            # update coefficients if rho has changed
            if self.rho != self.rho_old:
                self.rho_old = self.rho
                self.s = self.es.get_s(delta * self.rho)
                if self.s != self.s_old:
                    self.s_old = self.s
                    self.es.update_coefficients(self.s)

        self.last_rho_eval_counter += 1
