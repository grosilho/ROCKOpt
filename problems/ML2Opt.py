from jax import jit
from functools import partial

from problems.Opt.Problem import Problem as ProblemOpt


class ML2Opt(ProblemOpt):
    """
    A class to convert ./ML problems to a problem with the same interface as ./Opt problems.
    This is used to solve ./ML problems with the ./Opt solvers.
    """

    def __init__(self, problemML):
        self.problemML = problemML

    @partial(jit, static_argnums=(0,))
    def model_loss(self, params, batch_stats, inputs, labels):
        results = self.problemML.model.apply({'params': params, 'batch_stats': batch_stats}, x=inputs, train=False)
        loss = self.problemML.loss(results, labels, True)
        return loss

    @partial(jit, static_argnums=(0,))
    def model_loss_batch_stats(self, params, batch_stats, inputs, labels):
        results, updates = self.problemML.model.apply(
            {'params': params, 'batch_stats': batch_stats}, x=inputs, train=True, mutable=['batch_stats']
        )
        loss = self.problemML.loss(results, labels, True)
        return loss, updates["batch_stats"]

    def f(self, x):
        return self.model_loss(
            self.problemML.deflat_params(x),
            self.problemML.batch_stats,
            self.problemML.batch['inputs'],
            self.problemML.batch["labels"],
        )

    def update_params_batch_stats(self, x):
        self.problemML.params = self.problemML.deflat_params(x)
        loss, batch_stats = self.model_loss_batch_stats(
            self.problemML.params,
            self.problemML.batch_stats,
            self.problemML.batch['inputs'],
            self.problemML.batch["labels"],
        )
        self.problemML.batch_stats = batch_stats

    def initial_guess(self):
        return self.problemML.get_flattened_params_batch_stats()[0]

    def get_metrics_keys(self):
        return self.problemML.get_metrics_keys()

    def get_metrics(self, x):
        return self.problemML.get_metrics(self.problemML.deflat_params(x))
