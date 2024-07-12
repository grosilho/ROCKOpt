from jax import jit
from functools import partial
import tree_math as tm
from utils.HelperClass import HelperClass


class GD(HelperClass):
    def __init__(self, params):

        default_params = {"learning_rate": 1e-1}

        super().__init__(default_params, params)

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

    def step(self, problemML):

        loss, accuracy, batch_stats, grads = problemML.loss_accuracy_batch_stats_grads()
        problemML.params = self.body(problemML.params, self.learning_rate, grads).tree
        problemML.batch_stats = batch_stats
        self.stats["f_evals"] += 1
        self.stats["df_evals"] += 1

        metrics = {'batch_loss': loss, 'batch_accuracy': accuracy}
        accepted = True
        stepper_log_str = ""

        return metrics, accepted, stepper_log_str

    @partial(jit, static_argnums=(0,))
    def body(self, params, delta, grads):
        return tm.Vector(params) - delta * tm.Vector(grads)
