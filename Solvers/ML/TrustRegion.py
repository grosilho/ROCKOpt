from jax.numpy import abs
import time

from Problems.ML2Opt import ML2Opt
from Solvers.Opt.TrustRegion import TrustRegion as TrustRegionOpt


class TrustRegion(TrustRegionOpt):
    def __init__(self, problem, tensorboard_writer, **options):
        self.problemML = problem
        self.problemOpt = ML2Opt(self.problemML)
        self.tensorboard_writer = tensorboard_writer

        super().__init__(**options)

    def train(self, delta=None, max_iter=None):

        # Clean stats and history
        self.init_stats()
        self.init_history()

        if max_iter is None:
            max_iter = self.max_iter

        if delta is None:
            delta = self.delta_max

        record_rejected = False

        x, batch_stats = self.problemML.get_flattened_params_batch_stats()

        F = self.problemOpt

        et = time.time()

        i = 1
        n_batches = self.problemML.n_train_samples // self.problemML.batch_size

        while self.problemML.next_batch():

            if i == 1:
                fx = F.f(x)
                self.stats["f_evals"] += 1
                p = -F.df(x)
                self.stats["df_evals"] += 1
                self.append_to_history(x, fx, delta, True)

            p, gp, pBp = self.compute_direction(F, x, delta, p)

            rho, x, fx, accepted, norm_p, f_diff, model_diff = self.accept_or_reject_step(
                F, x, delta, fx, p, gp, pBp, record_rejected
            )

            self.log(x, fx, rho, delta, accepted, norm_p, f_diff, model_diff)

            delta = self.update_delta(f_diff, rho, norm_p, delta)

            metrics = self.problemML.train_metrics(self.problemML.deflat_params(x), batch_stats)
            epoch = i / n_batches
            self.logger.info(f"Epoch: {epoch:.2f}, Loss: {metrics['loss']:.3e}, Accuracy: {metrics['accuracy']:.3f}")

            if i % n_batches == 0:
                test_metrics = self.problemML.test_metrics(self.problemML.deflat_params(x), batch_stats)
                self.logger.info(
                    f"Epoch: {epoch:.2f}, Test Loss: {test_metrics['loss']:.3e}, Test Accuracy: {test_metrics['accuracy']:.3f}"
                )
                # self.tensorboard_writer.scalar("Loss/train", metrics['loss'], i)
                # self.tensorboard_writer.scalar("Accuracy/train", metrics['accuracy'], i)
                # self.tensorboard_writer.scalar("Loss/test", test_metrics['loss'], i)
                # self.tensorboard_writer.scalar("Accuracy/test", test_metrics['accuracy'], i)

            i += 1

            if self.check_convergence(abs(f_diff), norm_p, max_iter):
                break

            fx = F.loss_and_update_batch_stats(x)
            self.stats["f_evals"] += 1

        et = time.time() - et
        self.stats["cpu_time"] = et
        self.stats["cpu_time_per_iter"] = et / self.stats["iter"]

        self.stats["min_f"] = fx

        return self.history, self.stats
