import jax
import jax.numpy as jnp
from jax import jit
from functools import partial
import os
import matplotlib.pyplot as plt

from print_stuff import plot_options

from .Problem import Problem


class Huber(Problem):
    def __init__(self):

        m = 900
        d = 3000
        A_shape = (m, d)

        self.import_data(A_shape)

        # Regularization parameter
        self.delta = 1e-3
        self.lmbda = 0.2
        self.ell = 1e-2

        super().__init__('Huber')

    def import_data(self, A_shape):
        executed_file_dir = os.path.dirname(os.path.realpath(__file__))
        A_path = executed_file_dir + '/datasets/Huber_A.p'
        b_path = executed_file_dir + '/datasets/Huber_b.p'
        if not os.path.exists(A_path) or not os.path.exists(b_path):
            key = jax.random.key(1989)
            key1, key2 = jax.random.split(key)
            self.A = jax.random.normal(key1, A_shape) / jnp.sqrt(A_shape[1])
            self.b = jax.random.normal(key2, (A_shape[0],))
            jnp.save(A_path, self.A)
            jnp.save(b_path, self.b)
        else:
            self.A = jnp.load(A_path)
            self.b = jnp.load(b_path)

    @partial(jit, static_argnums=(0,))
    def f(self, x):
        return self.loss(x) + self.lmbda * self.huber(jnp.linalg.norm(x, ord=1)) + self.penalization(x)

    @partial(jit, static_argnums=(0,))
    def loss(self, x):
        return 0.5 * jnp.dot(self.A @ x - self.b, self.A @ x - self.b)

    @partial(jit, static_argnums=(0,))
    def penalization(self, x):
        return 0.5 * self.ell * jnp.dot(x, x)

    @partial(jit, static_argnums=(0,))
    def huber(self, x):
        return jnp.where(jnp.abs(x) <= self.delta, 0.5 * x**2, self.delta * (jnp.abs(x) - 0.5 * self.delta))

    def initial_guess(self):
        key = jax.random.key(2000)
        return jax.random.normal(key, (self.A.shape[1],))

    def plot(self, history):

        plt.set_loglevel(level='warning')
        figsize = (10, 5)
        fig = plt.figure(figsize=figsize)

        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        self.plot_fx_and_delta(ax1, ax2, history)

        plt.show()
