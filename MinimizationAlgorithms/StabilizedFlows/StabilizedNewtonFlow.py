import numpy as np
import logging
import time
from .es_methods import RKW1, RKC1, RKU1
from .rho_estimator import rho_estimator
from ..MinimizationAlgorithm import MinimizationAlgorithm


class StabilizedNewtonFlow(MinimizationAlgorithm):
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
        eps=1e-1,
        record_stages=False,
    ):

        description = "StabNF|" + method
        super().__init__(max_iter, atol, rtol, n, description)
        self.rho_freq = rho_freq
        self.method = method
        self.delta_max = delta_max
        self.es = eval(method)(damping, safe_add)
        self.rho_estimator = rho_estimator(2 * n)
        self.eps = eps
        self.record_stages = record_stages

        self.logger = logging.getLogger("StabilizedNewtonFlow")

    def solve(self, F, x, delta=None, max_iter=None, **kwargs):
        """
        Solves the minimization problem trying to follow the Newton direction.
        To do so we solve the ODE system
        x' = p, x(0) = x
        p' = (-B @ p - g)/eps, p(0) = -F.df(x)
        with a stabilized explicit method, where at the beginning of each time step we set
        B = F.ddf(x) (Hessian), g = F.df(x) (gradient) and eps is a small parameter.
        The p variable is the direction, which tends to the Newton direction, since the equilibrium of p' = (-B @ p - g)/eps is p = -B^-1 @ g.
        The smaller the eps the faster p converges to the Newton direction, but stiffness increases with eps->0.
        """

        self.init_stats()
        self.init_history()

        if max_iter is None:
            max_iter = self.max_iter

        if delta is None:
            delta = self.delta_max

        n = self.n

        rho_old = 0.0
        s_old = 0

        y = np.concatenate((x, -F.df(x)))
        self.stats["df_evals"] += 1

        self.append_to_history(x, F.f(x), delta, True)

        et = time.process_time()

        while True:

            g = F.df(x)
            self.stats["df_evals"] += 1

            def f(y):
                x = y[:n]
                p = y[n:]
                self.stats["ddf_mults"] += 1
                return np.concatenate((p, (-F.ddfv(x, p) - g) / self.eps))

            fy = f(y)

            # re-estimate rho every rho_freq iterations
            if self.stats["iter"] % self.rho_freq == 0:
                # rho, n_f_eval = self.rho_estimator.rho(f, y, fy)
                rho, n_f_eval = self.rho_estimator.rho_linear_power_method(lambda v: F.ddfv(x, v) / self.eps)

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
            dj = self.es.mu[0] * delta * fy
            for j in range(2, s + 1):
                if self.record_stages:
                    self.append_to_history(y + dj[:n], F.f(y + dj[:n]), delta, True)
                djm2, djm1, dj = djm1, dj, djm2
                dj = self.es.nu[j - 1] * djm1 + self.es.kappa[j - 1] * djm2 + self.es.mu[j - 1] * delta * f(y + djm1)

            y += dj
            x = y[:n]

            self.append_to_history(x, F.f(x), delta, True)

            f_diff = self.history["fx"][-1] - self.history["fx"][-2]
            if self.check_convergence(np.abs(f_diff), np.linalg.norm(dj[:n]), max_iter):
                break

        et = time.process_time() - et
        self.stats["cpu_time"] = et
        self.stats["min_f"] = F.f(x)
        self.stats["cpu_time_per_iter"] = et / self.stats["iter"]

        return self.history, self.stats
