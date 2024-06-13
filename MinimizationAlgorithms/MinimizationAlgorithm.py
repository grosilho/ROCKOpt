import numpy as np


class MinimizationAlgorithm:
    def __init__(self, max_iter, atol, rtol, n, description):
        self.max_iter = int(max_iter)
        self.atol = atol
        self.rtol = rtol
        self.n = n
        self.description = description

        self.history = {"x": [], "fx": [], "delta": [], "accepted": []}
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

    def append_to_history(self, x, fx, delta, accepted):
        self.history["x"].append(x)
        self.history["fx"].append(fx)
        self.history["delta"].append(delta)
        self.history["accepted"].append(accepted)

    def check_convergence(self):
        if len(self.history["x"]) < 2:
            return False
        fx_conv = np.abs(self.history["fx"][-1] - self.history["fx"][-2]) < np.max(
            [self.rtol * np.abs(self.history["fx"][-1]), self.atol]
        )
        x_conv = np.linalg.norm(self.history["x"][-1] - self.history["x"][-2]) < np.max(
            [self.rtol * np.linalg.norm(self.history["x"][-1]), self.atol]
        )

        return fx_conv or x_conv
