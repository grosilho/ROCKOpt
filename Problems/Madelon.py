import jax
import jax.numpy as jnp
from jax import jit
from functools import partial
import os
import pandas as pd

from .Problem import Problem


class Madelon(Problem):
    def __init__(self):

        # Identify the file type
        executed_file_dir = os.path.dirname(os.path.realpath(__file__))
        train_path = executed_file_dir + '/datasets/madelon_train.p'
        test_path = executed_file_dir + '/datasets/madelon_test.p'
        train_df = pd.read_pickle(train_path)
        test_df = pd.read_pickle(test_path)

        self.scale = 1e0

        self.train_features = jnp.array(train_df.drop('target', axis=1).to_numpy()) * self.scale
        self.train_targets = jnp.array(train_df['target'].to_numpy())
        self.test_features = jnp.array(test_df.drop('target', axis=1).to_numpy()) * self.scale
        self.test_targets = jnp.array(test_df['target'].to_numpy())

        self.tau = 5e3

        self.n_samples = self.train_features.shape[0]
        self.dim = self.train_features.shape[1]
        super().__init__('Madelon', self.dim)

        # print(train_df.loc[0, :])
        # print(train_features[0, :])
        # print(train_target[0])

    @partial(jit, static_argnums=(0,))
    def f(self, x):
        return self.f_train(x) + self.penalization(x)

    @partial(jit, static_argnums=(0,))
    def f_train(self, x):
        return jnp.mean(jax.vmap(self.f_loc, in_axes=[None, 0, 0])(x, self.train_features, self.train_targets))

    @partial(jit, static_argnums=(0,))
    def f_test(self, x):
        return jnp.mean(jax.vmap(self.f_loc, in_axes=[None, 0, 0])(x, self.test_features, self.test_targets))

    def f_loc(self, x, features, targets):
        return jnp.log(self.inv_expit(targets * jnp.dot(features, x)))

    def inv_expit(self, x):
        return 1.0 + jnp.exp(-x)

    def penalization(self, x):
        return 0.5 * self.tau * jnp.dot(x, x)

    def initial_guess(self):
        key = jax.random.key(1989)
        return jax.random.normal(key, shape=(self.dim,)) / 1e2
        # return jnp.zeros(self.dim)

    def rho(self, x):
        v = 1.0 / (1.0 + jnp.exp(self.train_targets * jnp.dot(self.train_features, x))) ** 2
        w = jnp.sum(self.train_features, axis=1)
        rho_f_train = jnp.max(jnp.dot(v * w, self.train_features)) / self.n_samples
        rho_penalization = self.tau
        return rho_f_train + rho_penalization

    def accuracy(self, x, features, targets):
        pred = jnp.sign(jnp.dot(features, x))
        return jnp.sum(abs(pred - targets) < 1e-2) / len(targets)

    def plot(self, history, plot_options):
        import matplotlib.pyplot as plt

        plt.set_loglevel(level='warning')
        figsize = (18, 8)
        fig = plt.figure(figsize=figsize)

        ax1 = fig.add_subplot(321)
        ax2 = fig.add_subplot(322)
        self.plot_fx_and_delta(ax1, ax2, history, plot_options)

        ax3 = fig.add_subplot(323)
        ax4 = fig.add_subplot(324)
        ax5 = fig.add_subplot(325)
        ax6 = fig.add_subplot(326)

        for key, val in history.items():
            x_hist = val["x"]
            f_train = [self.f_train(x) for x in x_hist]
            f_test = [self.f_test(x) for x in x_hist]
            accuracy_train = [self.accuracy(x, self.train_features, self.train_targets) for x in x_hist]
            accuracy_test = [self.accuracy(x, self.test_features, self.test_targets) for x in x_hist]
            iter = jnp.arange(0, len(x_hist))
            plot_opt_str = key.split("|")[0]
            ax3.semilogy(
                iter,
                f_train,
                marker=plot_options[plot_opt_str]["marker"],
                color=plot_options[plot_opt_str]["color"],
                fillstyle='none',
                label=key,
            )
            ax4.plot(
                iter,
                f_test,
                marker=plot_options[plot_opt_str]["marker"],
                color=plot_options[plot_opt_str]["color"],
                fillstyle='none',
                label=key,
            )
            ax5.plot(
                iter,
                accuracy_train,
                marker=plot_options[plot_opt_str]["marker"],
                color=plot_options[plot_opt_str]["color"],
                fillstyle='none',
                label=key,
            )
            ax6.plot(
                iter,
                accuracy_test,
                marker=plot_options[plot_opt_str]["marker"],
                color=plot_options[plot_opt_str]["color"],
                fillstyle='none',
                label=key,
            )
        ax3.set_title("Loss function (train)")
        ax4.set_title("Loss function (test)")
        ax5.set_title("Accuracy (train)")
        ax6.set_title("Accuracy (test)")
        ax3.legend()
        ax4.legend()
        ax5.legend()
        ax6.legend()

        plt.show()


if __name__ == "__main__":
    madelon = Madelon()
