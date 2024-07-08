from problems.ML2Opt import ML2Opt
from solvers.opt.TrustRegion import TrustRegion as TrustRegionOpt


class TrustRegionML(TrustRegionOpt):
    def __init__(self, problemML, tensorboard_writer, **options):
        self.problemML = problemML
        self.problemOpt = ML2Opt(self.problemML)
        self.tensorboard_writer = tensorboard_writer

        super().__init__(self.problemOpt, **options)

    def pre_iteration(self):
        return self.problemML.next_batch(self.stats["iter"])

    def post_iteration(self):
        self.fx = self.problemOpt.loss_and_update_batch_stats(self.x)
        self.stats["f_evals"] += 1

        epoch = self.stats["iter"] / self.problemML.n_batches
        train_metrics = self.problemML.train_metrics()
        test_metrics = self.problemML.test_metrics()
        self.logger.info(
            f"Epoch: {epoch:.2f}, Train Loss: {train_metrics['train_loss']:.3e}, Train Accuracy: {train_metrics['train_accuracy']:.3f}"
            + f", Test Loss: {test_metrics['test_loss']:.3e}, Test Accuracy: {test_metrics['test_accuracy']:.3f}"
        )
        if self.log_history:
            self.append_to_history(**train_metrics, **test_metrics)

        return True

    def log(self, x, fx, rho, delta, accepted, norm_p, f_diff, model_diff):
        super().log(x, fx, rho, delta, accepted, norm_p, f_diff, model_diff)
