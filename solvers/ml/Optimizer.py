import time

import jax.numpy as jnp
from flax.metrics import tensorboard

from utils.HelperClass import HelperClass


class Optimizer(HelperClass):
    def __init__(self, problemML, params, stepper_class, stepper_params):

        default_params = {
            "workdir": "tensorboard_logs",
            "max_iter": 10.0,
            "min_iter": 1,
            "rtol": 1e-6,
            "atol": 1e-6,
            "log_history": True,
            "record_rejected": False,
        }

        super().__init__(default_params, params)

        self.problemML = problemML
        self.stepper = stepper_class(stepper_params)
        self.tensorboard_writer = tensorboard.SummaryWriter(self.workdir)

        self.stats_keys = ["iter", "cpu_time", "cpu_time_per_iter"]
        self.history_keys = ["accepted"]

    def pre_loop(self):
        # Restart stats and history
        self.init_stats(self.stats_keys)
        self.init_history(self.history_keys + self.problemML.get_metrics_keys())

        params, batch_stats = self.problemML.init_params_batch_stats()
        self.loss = jnp.inf

        if self.log_history:
            self.append_to_history(**self.problemML.get_train_test_metrics())

        return params, batch_stats

    def solve(self):

        params, batch_stats = self.pre_loop()
        self.stepper.pre_loop(params, batch_stats)

        et = time.time()

        while self.problemML.next_batch(self.stats["iter"]):

            metrics, accepted, stepper_log_str = self.stepper.step(self.problemML)

            loss_diff = metrics['batch_loss'] - self.loss
            self.loss = metrics['batch_loss']

            self.stats["iter"] += 1
            epoch = self.stats["iter"] / self.problemML.n_batches

            self.log(epoch=epoch, stepper_log_str=stepper_log_str, loss_diff=loss_diff, **metrics)

            if self.log_history and (accepted or self.record_rejected):
                self.append_to_history(accepted=accepted, **metrics)

            if self.stats["iter"] % self.problemML.n_batches == 0:
                metrics = self.problemML.get_train_test_metrics()
                self.log(epoch=epoch, stepper_log_str="", **metrics)
                if self.log_history and (accepted or self.record_rejected):
                    self.append_to_history(**metrics)

                # self.tensorboard_writer.scalar("Loss/train", metrics['loss'], self.stats["iter"])
                # self.tensorboard_writer.scalar("Accuracy/train", metrics['accuracy'], self.stats["iter"])
                # self.tensorboard_writer.scalar("Loss/test", test_metrics['loss'], self.stats["iter"])
                # self.tensorboard_writer.scalar("Accuracy/test", test_metrics['accuracy'], self.stats["iter"])

            if self.converged(loss_diff, self.loss):
                break

        et = time.time() - et
        self.stats["cpu_time"] = et

        self.stepper.post_loop(params, batch_stats)
        history, stats = self.post_loop(params, batch_stats)

        return history, stats

    def post_loop(self, params, batch_stats):
        self.stats["cpu_time_per_iter"] = self.stats["cpu_time"] / self.stats["iter"]

        history = self.unify_histories(self.history, self.stepper.history)
        stats = self.add_stats(self.stats, self.stepper.stats)

        return history, stats

    def converged(self, f_diff, fx_norm):

        if self.stats["iter"] < self.min_iter:
            return False
        if self.stats["iter"] >= self.max_iter:
            self.logger.warning("Maximum number of iterations reached")
            return True

        abs_f_diff = jnp.abs(f_diff)
        fx_conv = abs_f_diff < self.rtol * fx_norm or abs_f_diff < self.atol

        if fx_conv:
            self.logger.debug("Convergence reached")
            return True

    def log(self, epoch, stepper_log_str, **metrics):
        metrics_str = ", ".join([f"{key.replace('_', ' ')}: {value:.3e}" for key, value in metrics.items()])
        self.logger.info(f"Epoch: {epoch:.2f}, " + metrics_str + ", " + stepper_log_str)
