import numpy as np
import logging


class MinimizationAlgorithm:
    def __init__(self, max_iter, atol, rtol, description):
        self.max_iter = int(max_iter)
        self.atol = atol
        self.rtol = rtol
        self.description = description

    def init_stats(self):
        self.stats = {
            "iter": 0,
            "cpu_time": 0.0,
            "cpu_time_per_iter": 0.0,
            "f_evals": 0,
            "df_evals": 0,
            "ddf_evals": 0,
            "ddf_solves": 0,
            "ddf_mults": 0,
            "min_f": 0.0,
        }

    def init_history(self):
        self.history = {"x": [], "fx": [], "delta": [], "accepted": []}

    def append_to_history(self, x, fx, delta, accepted):
        self.history["x"].append(x)
        self.history["fx"].append(fx)
        self.history["delta"].append(delta)
        self.history["accepted"].append(accepted)

    def check_convergence(self, f_diff, p_norm, max_iter):
        """
        Stopping criteria for the minimization algorithms.

        Arguments:
            - f_diff = f(x) - f(x_prev)
            - p_norm = ||x - x_prev||
            - max_iter = maximum number of iterations
        Returns:
            True if convergence is reached, False otherwise

        Stop when: the change in f is less than rtol * |f| + atol
                   or when the change in x is less than rtol * ||x|| + atol
                   or when the maximum number of iterations is reached
        """

        fx_conv = f_diff < np.max([self.rtol * np.abs(self.history["fx"][-1]), self.atol])
        x_conv = p_norm < np.max([self.rtol * np.linalg.norm(self.history["x"][-1]), self.atol])

        if fx_conv or x_conv:
            self.logger.debug("Convergence reached")
            return True
        elif self.stats["iter"] >= max_iter:
            self.logger.warning("Maximum number of iterations reached")
            return True

    def pre_iteration(self):
        pass

    def post_iteration(self):
        pass
