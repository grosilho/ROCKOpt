import numpy as np
import logging
import time
from .MinimizationAlgorithm import MinimizationAlgorithm


class ExactFlow(MinimizationAlgorithm):
    def __init__(self, max_iter, atol, rtol, n, delta_max):
        description = "Exact" + ("GF" if "Gradient" in self.__class__.__name__ else "NF")
        super().__init__(max_iter, atol, rtol, n, description)
        self.delta_max = delta_max

    def solve(self, F, x, delta=None, max_iter=None, **kwargs):
        """
        Solves very precisely the gradient or the newton flow, depending on the subclass.
        """
        self.init_stats()
        self.init_history()

        if max_iter is None:
            max_iter = self.max_iter

        if delta is None:
            delta = self.delta_max

        self.append_to_history(x, F.f(x), delta, True)

        et = time.process_time()

        while True:

            p = self.flow_direction(F, x)  # returns gradient or newton direction
            x += delta * p

            self.stats["iter"] += 1

            self.append_to_history(x, F.f(x), delta, True)

            f_diff = self.history["fx"][-1] - self.history["fx"][-2]
            if self.check_convergence(np.abs(f_diff), np.linalg.norm(p), max_iter):
                break

        et = time.process_time() - et
        self.stats["cpu_time"] = et
        self.stats["cpu_time_per_iter"] = et / self.stats["iter"]

        self.stats["min_f"] = self.history["fx"][-1]

        return self.history, self.stats


class ExactGradientFlow(ExactFlow):
    def __init__(self, max_iter, atol, rtol, n, delta_max):
        super().__init__(max_iter, atol, rtol, n, delta_max)
        self.logger = logging.getLogger("ExactGradientFlow")

    def flow_direction(self, F, x):
        self.stats["df_evals"] += 1
        return -F.df(x)


class ExactNewtonFlow(ExactFlow):
    def __init__(self, max_iter, atol, rtol, n, delta_max):
        super().__init__(max_iter, atol, rtol, n, delta_max)
        self.logger = logging.getLogger("ExactNewtonFlow")

    def flow_direction(self, F, x):
        self.stats["df_evals"] += 1
        self.stats["ddf_evals"] += 1
        self.stats["ddf_solves"] += 1
        return np.linalg.solve(F.ddf(x), -F.df(x))
