import numpy as np
import logging
import time
from scipy.sparse.linalg import LinearOperator
from .es_methods import RKW1, RKC1, RKU1
from .rho_estimator import rho_estimator
from ..MinimizationAlgorithm import MinimizationAlgorithm


class StabilizedGradientFlow(MinimizationAlgorithm):
    def __init__(
        self,
        max_iter,
        atol,
        rtol,
        delta_max=0.1,
        rho_freq=5,
        method="RKC1",
        damping=0.05,
        safe_add=0,
        record_stages=False,
    ):

        description = "StabGF|" + method
        super().__init__(max_iter, atol, rtol, description)

        self.record_stages = record_stages
        self.rho_freq = rho_freq
        self.method = method
        self.delta_max = delta_max
        self.es = eval(method)(damping, safe_add)
        self.rho_estimator = None

        self.logger = logging.getLogger("StabilizedGradientFlow")

    def solve(self, F, x, delta=None, max_iter=None, **kwargs):
        """
        Solve the gradient flow ODE
        x'= -F.df(x)
        using the stabilized method.
        """

        self.init_stats()
        self.init_history()

        if max_iter is None:
            max_iter = self.max_iter

        if delta is None:
            delta = self.delta_max

        def f(y):
            self.stats["df_evals"] += 1
            return -F.df(y)

        self.rho_old = 0.0
        self.s_old = 0

        if self.rho_estimator is None:
            self.rho_estimator = rho_estimator(x.size)

        self.append_to_history(x, F.f(x), delta, True)

        et = time.time()

        while True:
            fx = f(x)

            self.update_rho_and_stages(F, x, delta)

            self.stats["iter"] += 1

            x, norm_dx = self.step(f, x, delta, fx)

            Fx = F.f(x)
            self.append_to_history(x, Fx, delta, True)

            f_diff = self.history["fx"][-1] - self.history["fx"][-2]

            self.log(x, Fx, delta, norm_dx, f_diff)

            if self.check_convergence(np.abs(f_diff), norm_dx, max_iter):
                break

        et = time.time() - et
        self.stats["cpu_time"] = et
        self.stats["cpu_time_per_iter"] = et / self.stats["iter"]
        self.stats["min_f"] = F.f(x)

        return self.history, self.stats

    def update_rho_and_stages(self, F, x, delta):
        # re-estimate rho every rho_freq iterations
        if self.stats["iter"] % self.rho_freq == 0:
            # rho, n_f_eval = self.rho_estimator.rho(f, x, fx)
            self.rho, n_f_eval = self.rho_estimator.rho_linear_power_method(lambda v: F.ddfv(x, v))

        # update coefficients if rho has changed
        if self.rho != self.rho_old:
            self.rho_old = self.rho
            self.s = self.es.get_s(delta * self.rho)
            if self.s != self.s_old:
                self.s_old = self.s
                self.es.update_coefficients(self.s)

    def step(self, f, x, delta, fx):
        djm2, djm1 = 0.0, 0.0
        dj = self.es.mu[0] * delta * fx
        for j in range(2, self.s + 1):
            djm2, djm1, dj = djm1, dj, djm2
            dj = self.es.nu[j - 1] * djm1 + self.es.kappa[j - 1] * djm2 + self.es.mu[j - 1] * delta * f(x + djm1)

        x += dj
        norm_dj = np.linalg.norm(dj)

        return x, norm_dj

    def log(self, x, fx, delta, norm_dx, f_diff):
        self.logger.info(
            f"Iteration {self.stats["iter"]}: {(f'x = {x.ravel()}, ' if x.size < 5 else "")}f(x) = {fx:.3e}, eigval = {self.rho:.3e}, {u"Δ"} = {delta:.3e}, s = {self.s}"
            + f", dx = {norm_dx:.3e}, df = {f_diff:.3e}"
        )
