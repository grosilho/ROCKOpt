from jax.flatten_util import ravel_pytree
from problems.ML2Opt import ML2Opt
from solvers.opt.ExactFlow import ExactGradientFlow as ExactGradientFlowOpt


class ExactGradientFlowML(ExactGradientFlowOpt):
    def __init__(self, problemML, tensorboard_writer, **options):
        self.problemML = problemML
        self.problemOpt = ML2Opt(self.problemML)
        self.tensorboard_writer = tensorboard_writer

        super().__init__(self.problemOpt, **options)

    def flow_direction(self, x):
        self.problemML.params = self.problemML.deflat_params(x)
        loss, accuracy, batch_stats, grads = self.problemML.loss_accuracy_batch_stats_grads()
        self.problemML.batch_stats = batch_stats
        return -ravel_pytree(grads)[0]

    def pre_iteration(self):
        return self.problemML.next_batch(self.stats["iter"])

    def post_iteration(self):

        self.problemOpt.update_params_batch_stats(self.x)

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
