from jax.flatten_util import ravel_pytree
import jax.numpy as jnp
import optax
import time

from solvers.opt.MinimizationAlgorithm import MinimizationAlgorithm


class OptaxOptimizer(MinimizationAlgorithm):
    def __init__(self, problemML, tensorboard_writer, **options):
        self.problemML = problemML
        self.tensorboard_writer = tensorboard_writer

        super().__init__(**options)

    def create_state(self):
        params, batch_stats = self.problemML.get_params_batch_stats()
        state = self.tx.init(params)
        return state

    def train_step(self, state):
        """Train for a single step."""
        loss, accuracy, batch_stats, grads = self.problemML.loss_accuracy_batch_stats_grads()
        updates, state = self.tx.update(grads, state)

        self.problemML.params = optax.apply_updates(self.problemML.params, updates)
        self.problemML.batch_stats = batch_stats

        metrics = {'batch_loss': loss, 'batch_accuracy': accuracy}
        self.stats["f_evals"] += 1
        self.stats["df_evals"] += 1

        return state, metrics, updates

    def solve(self, max_iter=None, min_iter=None):

        if max_iter is None:
            max_iter = self.max_iter
        if min_iter is None:
            min_iter = self.min_iter

        state = self.create_state()
        self.init_stats()
        self.init_history(self.problemML.get_metrics_keys())

        et = time.time()

        while self.problemML.next_batch(self.stats["iter"]):

            if self.stats["iter"] == 0:
                metrics = self.problemML.train_metrics()
                loss = metrics['train_loss']
                self.stats["f_evals"] += 1

            state, metrics, updates = self.train_step(state)

            prev_loss = loss
            loss = metrics['batch_loss']
            loss_diff = prev_loss - loss

            if self.log_history:
                self.append_to_history(**metrics)

            updates_norm = jnp.linalg.norm(ravel_pytree(updates)[0])
            params_norms = jnp.linalg.norm(ravel_pytree(self.problemML.params)[0])

            epoch = (self.stats["iter"] + 1) / self.problemML.n_batches
            self.logger.info(
                f"Epoch: {epoch:.2f}, Batch Loss: {metrics['batch_loss']:.3e}, Batch Accuracy: {metrics['batch_accuracy']:.3f}"
                + f", dx = {updates_norm:.3e}, df = {loss_diff:.3e}"
            )

            if self.stats["iter"] % self.problemML.n_batches == 0:
                train_metrics = self.problemML.train_metrics()
                test_metrics = self.problemML.test_metrics()
                self.logger.info(
                    f"Epoch: {epoch:.2f}, Train Loss: {train_metrics['train_loss']:.3e}, Train Accuracy: {train_metrics['train_accuracy']:.3f}"
                    + f", Test Loss: {test_metrics['test_loss']:.3e}, Test Accuracy: {test_metrics['test_accuracy']:.3f}"
                )
                if self.log_history:
                    self.append_to_history(**train_metrics, **test_metrics)

                # self.tensorboard_writer.scalar("Loss/train", metrics['loss'], self.stats["iter"])
                # self.tensorboard_writer.scalar("Accuracy/train", metrics['accuracy'], self.stats["iter"])
                # self.tensorboard_writer.scalar("Loss/test", test_metrics['loss'], self.stats["iter"])
                # self.tensorboard_writer.scalar("Accuracy/test", test_metrics['accuracy'], self.stats["iter"])

            self.stats["iter"] += 1

            if self.check_convergence(
                jnp.abs(loss_diff), updates_norm, params_norms, jnp.abs(loss), max_iter, min_iter
            ):
                break

        et = time.time() - et
        self.stats["cpu_time"] = et
        self.stats["cpu_time_per_iter"] = et / self.stats["iter"]

        return self.history, self.stats


class SGD(OptaxOptimizer):
    def __init__(self, problemML, tensorboard_writer, learning_rate, **options):
        description = "SGD"
        super().__init__(problemML, tensorboard_writer, description=description, **options)

        # momentum = 0.0
        self.tx = optax.sgd(learning_rate)


class Adam(OptaxOptimizer):
    def __init__(self, problemML, tensorboard_writer, learning_rate, **options):
        description = "Adam"
        super().__init__(problemML, tensorboard_writer, description=description, **options)

        b1 = 0.9
        b2 = 0.999
        eps = 1e-8
        eps_root = 1e-8
        self.tx = optax.adam(learning_rate, b1=b1, b2=b2, eps=eps, eps_root=eps_root)
