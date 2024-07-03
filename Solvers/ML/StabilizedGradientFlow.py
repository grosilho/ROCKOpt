import numpy as np
import time

from Problems.ML2Opt import ML2Opt
from Solvers.Opt.StabilizedFlows.StabilizedGradientFlow import StabilizedGradientFlow as StabilizedGradientFlowOpt
from Solvers.Opt.StabilizedFlows.rho_estimator import rho_estimator


class StabilizedGradientFlow(StabilizedGradientFlowOpt):
    def __init__(self, problem, tensorboard_writer, **options):
        self.problemML = problem
        self.problemOpt = ML2Opt(self.problemML)
        self.tensorboard_writer = tensorboard_writer

        super().__init__(**options)

    def train(self, delta=None, max_iter=None):
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

        x, batch_stats = self.problemML.get_flattened_params_batch_stats()

        F = self.problemOpt

        def f(y):
            self.stats["df_evals"] += 1
            return -F.df(y)

        self.rho_old = 0.0
        self.s_old = 0

        if self.rho_estimator is None:
            self.rho_estimator = rho_estimator(x.size)

        i = 1
        n_batches = self.problemML.n_train_samples // self.problemML.batch_size

        et = time.time()

        while self.problemML.next_batch():

            if i == 1:
                self.append_to_history(x, F.f(x), delta, True)
                self.stats["f_evals"] += 1

            fx = f(x)

            self.update_rho_and_stages(F, x, delta)

            self.stats["iter"] += 1

            x, norm_dx = self.step(f, x, delta, fx)

            Fx = F.f(x)
            self.append_to_history(x, Fx, delta, True)

            f_diff = self.history["fx"][-1] - self.history["fx"][-2]

            self.log(x, Fx, delta, norm_dx, f_diff)

            metrics = self.problemML.train_metrics(self.problemML.deflat_params(x), batch_stats)
            epoch = i / n_batches
            self.logger.info(f"Epoch: {epoch:.2f}, Loss: {metrics['loss']:.3e}, Accuracy: {metrics['accuracy']:.3f}")

            if self.check_convergence(np.abs(f_diff), norm_dx, max_iter):
                break

            i += 1

            F.loss_and_update_batch_stats(x)
            self.stats["f_evals"] += 1

        et = time.time() - et
        self.stats["cpu_time"] = et
        self.stats["cpu_time_per_iter"] = et / self.stats["iter"]
        self.stats["min_f"] = F.f(x)

        return self.history, self.stats
