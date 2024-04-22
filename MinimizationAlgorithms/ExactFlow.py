import numpy as np
import logging
import time
from .MinimizationAlgorithm import MinimizationAlgorithm


class ExactFlow(MinimizationAlgorithm):
    def __init__(self, max_iter, tol, n, delta_max):
        description = "Exact" + ("GF" if "Gradient" in self.__class__.__name__ else "NF")
        super().__init__(max_iter, tol, n, description)
        self.delta_max = delta_max

    def solve(self, F, x, delta=0.0, **kwargs):
        """
        Solves very precisely the gradient or the newton flow, depending on the subclass.
        """
        if delta == 0.0:
            delta = self.delta_max

        self.append_to_history(x, delta, True)

        et = time.process_time()

        while True:

            p = self.flow_direction(F, x)  # returns gradient or newton direction
            x += delta * p

            self.stats["iter"] += 1

            self.append_to_history(x, delta, True)

            if np.linalg.norm(p) / np.sqrt(self.n) < self.tol:
                self.logger.debug("Convergence reached")
                break

            if self.stats["iter"] >= self.max_iter:
                self.logger.warning("Maximum number of iterations reached in exact flow computation.")
                break

        et = time.process_time() - et
        self.stats["cpu_time"] = et

        return self.history, self.stats


class ExactGradientFlow(ExactFlow):
    def __init__(self, max_iter, tol, n, delta_max):
        super().__init__(max_iter, tol, n, delta_max)
        self.logger = logging.getLogger("ExactGradientFlow")

    def flow_direction(self, F, x):
        self.stats["df_evals"] += 1
        return -F.df(x)


class ExactNewtonFlow(ExactFlow):
    def __init__(self, max_iter, tol, n, delta_max):
        super().__init__(max_iter, tol, n, delta_max)
        self.logger = logging.getLogger("ExactNewtonFlow")

    def flow_direction(self, F, x):
        self.stats["df_evals"] += 1
        self.stats["ddf_evals"] += 1
        self.stats["ddf_solves"] += 1
        return np.linalg.solve(F.ddf(x), -F.df(x))
