import numpy as np
import logging
import time
from .es_methods import RKW1, RKC1, RKU1
from .rho_estimator import rho_estimator
from ..MinimizationAlgorithm import MinimizationAlgorithm


class StabilizedGradientFlow(MinimizationAlgorithm):
    def __init__(
        self,
        max_iter,
        atol,
        rtol,
        n,
        delta_max=0.1,
        rho_freq=5,
        method="RKC1",
        damping=0.05,
        safe_add=0,
        record_stages=False,
    ):

        description = "StabGF|" + method
        super().__init__(max_iter, atol, rtol, n, description)

        self.record_stages = record_stages
        self.rho_freq = rho_freq
        self.method = method
        self.delta_max = delta_max
        self.es = eval(method)(damping, safe_add)
        self.rho_estimator = rho_estimator(n)

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

        rho_old = 0.0
        s_old = 0

        self.append_to_history(x, F.f(x), delta, True)

        et = time.process_time()

        while True:
            fx = f(x)

            # re-estimate rho every rho_freq iterations
            if self.stats["iter"] % self.rho_freq == 0:
                if not hasattr(F, "rho"):
                    rho, n_f_eval = self.rho_estimator.rho(f, x, fx)
                else:
                    rho = F.rho(x)

            # update coefficients if rho has changed
            if rho != rho_old:
                rho_old = rho
                s = self.es.get_s(delta * rho)
                if s != s_old:
                    s_old = s
                    self.es.update_coefficients(s)

            self.stats["iter"] += 1
            self.logger.info(
                f"Iteration {self.stats["iter"]}: {(f'x = {x.ravel()}, ' if x.size < 5 else "")}f(x) = {F.f(x):.3e}, eigval = {rho:.3e}, {u"Δ"} = {delta:.3e}, s = {s}"
            )

            djm2, djm1 = 0.0, 0.0
            dj = self.es.mu[0] * delta * fx
            for j in range(2, s + 1):
                if self.record_stages:
                    self.append_to_history(x + dj, F.f(x + dj), delta, True)
                djm2, djm1, dj = djm1, dj, djm2
                dj = self.es.nu[j - 1] * djm1 + self.es.kappa[j - 1] * djm2 + self.es.mu[j - 1] * delta * f(x + djm1)

            x += dj

            self.append_to_history(x, F.f(x), delta, True)

            f_diff = self.history["fx"][-1] - self.history["fx"][-2]
            if self.check_convergence(np.abs(f_diff), np.linalg.norm(dj), max_iter):
                break

        et = time.process_time() - et
        self.stats["cpu_time"] = et
        self.stats["cpu_time_per_iter"] = et / self.stats["iter"]
        self.stats["min_f"] = F.f(x)

        return self.history, self.stats
