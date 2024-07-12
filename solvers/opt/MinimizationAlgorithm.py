import time

import jax.numpy as jnp

from utils.HelperClass import HelperClass


class MinimizationAlgorithm(HelperClass):
    def __init__(self, problem, params, stepper_class, stepper_params):

        default_params = {
            "max_iter": 1000,
            "min_iter": 1,
            "rtol": 1e-3,
            "atol": 0.0,
            "log_history": False,
            "record_rejected": False,
        }
        super().__init__(default_params, params)

        self.problem = problem
        self.stepper = stepper_class(problem, stepper_params)

        self.stats_keys = [
            "iter",
            "cpu_time",
            "cpu_time_per_iter",
            "f_evals",
            "df_evals",
            "ddf_evals",
            "ddf_solves",
            "ddf_mults",
            "f(x)",
        ]
        self.history_keys = ["fx", "accepted"]

    def pre_loop(self):
        # Restart stats and history
        self.init_stats(self.stats_keys)
        self.init_history(self.history_keys + self.problem.get_metrics_keys())

        x = self.problem.initial_guess(self.stepper.dtype)
        fx = self.problem.f(x)
        self.stats["f_evals"] += 1

        self.log(fx, "")

        if self.log_history:
            self.append_to_history(fx=fx, **self.problem.get_metrics(x))

        return x, fx

    def solve(self):

        x, fx = self.pre_loop()
        self.stepper.pre_loop(x, fx)

        et = time.time()

        while True:

            self.stats["iter"] += 1

            x, fx, accepted, convergence_info, stepper_log_str = self.stepper.step(x, fx)

            self.log(fx, stepper_log_str)

            if self.log_history and (accepted or self.record_rejected):
                self.append_to_history(fx=fx, accepted=accepted, **self.problem.get_metrics(x))

            if self.converged(**convergence_info):
                break

        et = time.time() - et
        self.stats["cpu_time"] = et

        self.stepper.post_loop(x, fx)
        history, stats = self.post_loop(x, fx)

        return history, stats

    def post_loop(self, x, fx):
        self.stats["cpu_time_per_iter"] = self.stats["cpu_time"] / self.stats["iter"]
        self.stats["f(x)"] = fx

        history = self.unify_histories(self.history, self.stepper.history)
        stats = self.add_stats(self.stats, self.stepper.stats)

        return history, stats

    def converged(self, f_diff, dx_norm, x_norm, fx_norm):
        """
        Stopping criteria for the minimization algorithms.

        Arguments:
            - f_diff = f(x) - f(x_prev)
            - p_norm = ||x - x_prev||
            - x_norm = ||x||
            - fx_norm = |f(x)|
            - max_iter = maximum number of iterations
        Returns:
            True if convergence is reached, False otherwise

        """

        if self.stats["iter"] < self.min_iter:
            return False
        if self.stats["iter"] >= self.max_iter:
            self.logger.warning("Maximum number of iterations reached")
            return True

        abs_f_diff = jnp.abs(f_diff)
        fx_conv = abs_f_diff < self.rtol * fx_norm or abs_f_diff < self.atol
        x_conv = dx_norm < self.rtol * x_norm or dx_norm < self.atol

        if fx_conv or x_conv:
            self.logger.debug("Convergence reached")
            return True

    def log(self, fx, stepper_log_str):
        self.logger.info(f"Iteration {self.stats['iter']}, f(x) = {fx:.3e}, " + stepper_log_str)
