from jax.flatten_util import ravel_pytree
import jax.numpy as jnp
import optax

from utils.HelperClass import HelperClass


class OptaxOptimizer(HelperClass):
    def __init__(self, params):

        default_params = {
            "optax_optimizer_class": optax.sgd,
            "optax_optimizer_params": {"learning_rate": 1e-1},
        }
        super().__init__(default_params, params)
        self.optimizer = self.optax_optimizer_class(**self.optax_optimizer_params)

        self.stats_keys = [
            "f_evals",
            "df_evals",
            "ddf_evals",
            "ddf_solves",
            "ddf_mults",
        ]
        self.history_keys = []

    def pre_loop(self, params, batch_stats):
        self.init_stats(self.stats_keys)
        self.init_history(self.history_keys)
        self.state = self.optimizer.init(params)

    def step(self, problemML):
        """Train for a single step."""
        loss, accuracy, batch_stats, grads = problemML.loss_accuracy_batch_stats_grads()
        updates, self.state = self.optimizer.update(grads, self.state)

        problemML.params = optax.apply_updates(problemML.params, updates)
        problemML.batch_stats = batch_stats

        metrics = {'batch_loss': loss, 'batch_accuracy': accuracy}
        self.stats["f_evals"] += 1
        self.stats["df_evals"] += 1

        updates_norm = jnp.linalg.norm(ravel_pytree(updates)[0])

        return metrics, updates_norm
