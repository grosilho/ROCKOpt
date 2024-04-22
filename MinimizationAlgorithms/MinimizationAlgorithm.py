import numpy as np


class MinimizationAlgorithm:
    def __init__(self, max_iter, tol, n, description):
        self.max_iter = int(max_iter)
        self.tol = tol
        self.n = n
        self.description = description

        self.history = {"x": [], "delta": [], "accepted": []}
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

    def append_to_history(self, x, delta, accepted):
        self.history["x"].append(x)
        self.history["delta"].append(delta)
        self.history["accepted"].append(accepted)
