import jax
import jax.numpy as jnp
from jax import jit, lax
from jax.flatten_util import ravel_pytree
from flax import linen as nn
from functools import partial
import os
import pandas as pd

from .Problem import Problem


class Madelon(Problem):
    def __init__(self):

        # model widths
        self.widths = [250, 1]
        # Regularization parameter
        self.tau = 0.01

        super().__init__('Madelon')

    def import_data(self):
        executed_file_dir = os.path.dirname(os.path.realpath(__file__))
        train_path = executed_file_dir + '/datasets/madelon_train.p'
        test_path = executed_file_dir + '/datasets/madelon_test.p'
        train_df = pd.read_pickle(train_path)
        test_df = pd.read_pickle(test_path)

        self.train_features = jnp.array(train_df.drop('target', axis=1).to_numpy())
        self.test_features = jnp.array(test_df.drop('target', axis=1).to_numpy())
        self.train_targets = jnp.array(train_df['target'].to_numpy()).reshape((-1, 1))
        self.test_targets = jnp.array(test_df['target'].to_numpy()).reshape((-1, 1))

        # Convert -1, 1 to 0, 1
        self.train_targets = (self.train_targets + 1.0) / 2.0
        self.test_targets = (self.test_targets + 1.0) / 2.0
        # Standardize the data
        mu = jnp.mean(jnp.vstack((self.train_features, self.test_features)), axis=0)
        sigma = jnp.std(jnp.vstack((self.train_features, self.test_features)), axis=0)
        self.train_features = (self.train_features - mu) / sigma
        self.test_features = (self.test_features - mu) / sigma

        self.n_samples = self.train_features.shape[0]

    def define_model(self, mp_dtype):

        def model(dtype):
            class ExplicitMLP(nn.Module):
                features: list[int]

                def setup(self):
                    self.layers = [nn.Dense(feat, dtype=dtype) for feat in self.features]

                def __call__(self, inputs):
                    x = inputs
                    for i, lyr in enumerate(self.layers):
                        x = lyr(x)
                        if i != len(self.layers) - 1:
                            x = nn.relu(x)
                    return x

            return ExplicitMLP(self.widths)

        self.model = dict()
        self.model[mp_dtype.high.dtype] = model(mp_dtype.high.dtype)
        if mp_dtype.low is not None:
            self.model[mp_dtype.low.dtype] = model(mp_dtype.low.dtype)

    @partial(jit, static_argnums=(0,))
    def f(self, x):
        return self.f_train(x) + self.penalization(x)

    @partial(jit, static_argnums=(0,))
    def f_train(self, x):
        return jnp.mean(self.cross_entropy(x, self.train_features[x.dtype], self.train_targets[x.dtype]))

    @partial(jit, static_argnums=(0,))
    def f_test(self, x):
        return jnp.mean(self.cross_entropy(x, self.test_features[x.dtype], self.test_targets[x.dtype]))

    @partial(jit, static_argnums=(0,))
    def cross_entropy(self, x, features, targets):
        model_output = self.model[x.dtype].apply(self.deflat_params(x), features)
        return -targets * jnp.log(self.expit(model_output)) - (1.0 - targets) * jnp.log(1.0 - self.expit(model_output))

    @partial(jit, static_argnums=(0,))
    def expit(self, x):
        return 1.0 / (1.0 + jnp.exp(-x))

    @partial(jit, static_argnums=(0,))
    def penalization(self, x):
        return 0.5 * self.tau * jnp.dot(x, x)

    def initial_guess(self, mp_dtype):
        self.import_data()
        self.train_features = {mp_dtype.high.dtype: lax.convert_element_type(self.train_features, mp_dtype.high)}
        self.train_targets = {mp_dtype.high.dtype: lax.convert_element_type(self.train_targets, mp_dtype.high)}
        self.test_features = {mp_dtype.high.dtype: lax.convert_element_type(self.test_features, mp_dtype.high)}
        self.test_targets = {mp_dtype.high.dtype: lax.convert_element_type(self.test_targets, mp_dtype.high)}
        if mp_dtype.low is not None:
            self.train_features[mp_dtype.low.dtype] = lax.convert_element_type(
                self.train_features[mp_dtype.high.dtype], mp_dtype.low
            )
            self.train_targets[mp_dtype.low.dtype] = lax.convert_element_type(
                self.train_targets[mp_dtype.high.dtype], mp_dtype.low
            )
            self.test_features[mp_dtype.low.dtype] = lax.convert_element_type(
                self.test_features[mp_dtype.high.dtype], mp_dtype.low
            )
            self.test_targets[mp_dtype.low.dtype] = lax.convert_element_type(
                self.test_targets[mp_dtype.high.dtype], mp_dtype.low
            )
        self.define_model(mp_dtype)
        key = jax.random.key(1989)
        params = self.model[mp_dtype.high.dtype].init(key, self.train_features[mp_dtype.high.dtype][0])
        flatened_params, self.deflat_params = ravel_pytree(params)
        return flatened_params

    def accuracy(self, x, features, targets):
        pred = jnp.round(self.expit(self.model[x.dtype].apply(self.deflat_params(x), features)))
        return jnp.sum(abs(pred - targets) < 1e-2) / len(targets)

    def get_metrics(self, x):
        return {
            "train_accuracy": self.accuracy(x, self.train_features[x.dtype], self.train_targets[x.dtype]),
            "test_accuracy": self.accuracy(x, self.test_features[x.dtype], self.test_targets[x.dtype]),
            "f_train": self.f_train(x),
            "f_test": self.f_test(x),
        }

    def get_metrics_keys(self):
        return ["train_accuracy", "test_accuracy", "f_train", "f_test"]

    def plot(self, histories):
        import matplotlib.pyplot as plt

        plt.set_loglevel(level='warning')
        figsize = (18, 8)
        fig = plt.figure(figsize=figsize)

        axes = fig.subplots(nrows=3, ncols=2)
        self.plot_histories(
            histories=histories,
            keys=["fx", "delta", "f_train", "f_test", "train_accuracy", "test_accuracy"],
            axes=axes.ravel(),
        )

        plt.show()
