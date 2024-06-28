import jax
import jax.numpy as jnp
from jax import jit
from jax.flatten_util import ravel_pytree
from flax import linen as nn
from functools import partial
import os
import pandas as pd

from print_stuff import plot_options

from .Problem import Problem


class Madelon(Problem):
    def __init__(self):

        widths = [250, 1]
        self.import_data()
        self.define_model(widths)

        # Regularization parameter
        self.tau = 1e-2

        self.n_samples = self.train_features.shape[0]
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

    def define_model(self, widths):
        class ExplicitMLP(nn.Module):
            features: list[int]

            def setup(self):
                self.layers = [nn.Dense(feat) for feat in self.features]

            def __call__(self, inputs):
                x = inputs
                for i, lyr in enumerate(self.layers):
                    x = lyr(x)
                    if i != len(self.layers) - 1:
                        x = nn.relu(x)
                return x

        self.model = ExplicitMLP(features=widths)

    @partial(jit, static_argnums=(0,))
    def f(self, x):
        return self.f_train(x) + self.penalization(x)

    @partial(jit, static_argnums=(0,))
    def f_train(self, x):
        return jnp.mean(self.cross_entropy(x, self.train_features, self.train_targets))

    @partial(jit, static_argnums=(0,))
    def f_test(self, x):
        return jnp.mean(self.cross_entropy(x, self.train_features, self.train_targets))

    @partial(jit, static_argnums=(0,))
    def cross_entropy(self, x, features, targets):
        return -targets * jnp.log(self.expit(self.model.apply(self.deflat_params(x), features))) - (
            1.0 - targets
        ) * jnp.log(1.0 - self.expit(self.model.apply(self.deflat_params(x), features)))

    @partial(jit, static_argnums=(0,))
    def expit(self, x):
        return 1.0 / (1.0 + jnp.exp(-x))

    @partial(jit, static_argnums=(0,))
    def penalization(self, x):
        return 0.5 * self.tau * jnp.dot(x, x)

    def initial_guess(self):
        key = jax.random.key(1989)
        params = self.model.init(key, self.train_features[0])
        flatened_params, self.deflat_params = ravel_pytree(params)
        return flatened_params

    def accuracy(self, x, features, targets):
        pred = jnp.round(self.expit(self.model.apply(self.deflat_params(x), features)))
        return jnp.sum(abs(pred - targets) < 1e-2) / len(targets)

    def plot(self, history):
        import matplotlib.pyplot as plt

        plt.set_loglevel(level='warning')
        figsize = (18, 8)
        fig = plt.figure(figsize=figsize)

        ax1 = fig.add_subplot(321)
        ax2 = fig.add_subplot(322)
        # Loss function + penalization and delta
        self.plot_fx_and_delta(ax1, ax2, history)

        # Loss function without penalization for train and test
        # Accuracy for train and test
        ax3 = fig.add_subplot(323)
        ax4 = fig.add_subplot(324)
        ax5 = fig.add_subplot(325)
        ax6 = fig.add_subplot(326)

        for key, val, color, marker in zip(history.keys(), history.values(), *plot_options()):
            x_hist = val["x"]
            f_train = [self.f_train(x) for x in x_hist]
            f_test = [self.f_test(x) for x in x_hist]
            accuracy_train = [self.accuracy(x, self.train_features, self.train_targets) for x in x_hist]
            accuracy_test = [self.accuracy(x, self.test_features, self.test_targets) for x in x_hist]
            iter = jnp.arange(0, len(x_hist))
            ax3.semilogy(iter, f_train, marker=marker, color=color, fillstyle='none', label=key)
            ax4.plot(iter, f_test, marker=marker, color=color, fillstyle='none', label=key)
            ax5.plot(iter, accuracy_train, marker=marker, color=color, fillstyle='none', label=key)
            ax6.plot(iter, accuracy_test, marker=marker, color=color, fillstyle='none', label=key)

        ax3.set_title("Loss function (train)")
        ax4.set_title("Loss function (test)")
        ax5.set_title("Accuracy (train)")
        ax6.set_title("Accuracy (test)")
        ax3.legend()
        ax4.legend()
        ax5.legend()
        ax6.legend()

        plt.show()
