import numpy as np
import jax
import jax.numpy as jnp
from jax import jit
from functools import partial
import os
import matplotlib.pyplot as plt
from PIL import Image

from .Problem import Problem


class Denoising(Problem):
    def __init__(self):

        self.grayscale = False
        self.max_size = 400

        self.import_data()
        self.reformat_data()

        self.lmbda = 0.5
        self.eps = 1e-4

        super().__init__('Denoising')

    def import_data(self):
        executed_file_dir = os.path.dirname(os.path.realpath(__file__))
        path = executed_file_dir + '/datasets/graffiti.jpg'
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        self.img_clean = Image.open(path)
        if self.grayscale:
            self.img_clean = self.img_clean.convert('L')

        self.img_clean.thumbnail((self.max_size, self.max_size))
        self.img_clean.save(path.replace('.jpg', '_thumbnail.jpg'))

        self.arr_clean = jnp.array(self.img_clean)
        noise = 40.0 * jax.random.normal(jax.random.key(1989), shape=self.arr_clean.shape)
        self.arr_noised = self.arr_clean + noise
        self.arr_noised = jnp.clip(self.arr_noised, 0, 255).astype(jnp.uint8)
        self.img_noised = Image.fromarray(np.asarray(self.arr_noised))
        self.img_noised.save(path.replace('.jpg', '_thumbnail_noised.jpg'))

    def reformat_data(self):
        self.shape = self.arr_clean.shape
        self.size = self.arr_clean.size

        def reformat(a):
            a = a.astype(jnp.float32)
            a = a / 255.0
            return a.ravel()

        self.arr_clean = reformat(self.arr_clean)
        self.arr_noised = reformat(self.arr_noised)

    @partial(jit, static_argnums=(0,))
    def f(self, x):
        return self.loss(x) + self.penalization(x)

    @partial(jit, static_argnums=(0,))
    def loss(self, x):
        return 0.5 * jnp.dot(self.arr_noised - x, self.arr_noised - x) / self.size

    @partial(jit, static_argnums=(0,))
    def penalization(self, x):
        return self.lmbda * jnp.mean(self.norm_eps(self.grad_img(x)))

    @partial(jit, static_argnums=(0,))
    def norm_eps(self, grad):
        return jnp.sqrt(grad[0] ** 2 + grad[1] ** 2 + self.eps**2)

    @partial(jit, static_argnums=(0,))
    def grad_img(self, x):
        xr = x.reshape(self.shape)
        grad1 = jnp.diff(xr, axis=0)
        grad1 = (grad1[:-1, :] + grad1[1:, :]) / 2.0
        pad = [[1, 1], [0, 0]] + [[0, 0]] * (len(grad1.shape) - 2)
        grad1 = jnp.pad(grad1, pad, mode='edge')
        grad2 = jnp.diff(xr, axis=1)
        grad2 = (grad2[:, :-1] + grad2[:, 1:]) / 2.0
        pad = [[0, 0], [1, 1]] + [[0, 0]] * (len(grad1.shape) - 2)
        grad2 = jnp.pad(grad2, pad, mode='edge')
        return grad1, grad2

    def initial_guess(self):
        return self.arr_noised

    def plot(self, history):

        plt.set_loglevel(level='warning')

        n_methods = len(history)
        n_lines = 2 + n_methods // 2 + n_methods % 2

        figsize = (n_lines * 8, 10)
        fig = plt.figure(figsize=figsize)

        ax1 = fig.add_subplot(n_lines, 2, 1)
        ax2 = fig.add_subplot(n_lines, 2, 2)
        self.plot_fx_and_delta(ax1, ax2, history)

        ax3 = fig.add_subplot(n_lines, 2, 3)
        ax4 = fig.add_subplot(n_lines, 2, 4)

        ax3.imshow(self.arr_clean.reshape(self.shape), cmap='gray')
        ax4.imshow(self.arr_noised.reshape(self.shape), cmap='gray')
        ax3.set_title("Original")
        ax4.set_title("Noised")

        for i, (method, data) in enumerate(history.items()):
            ax = fig.add_subplot(n_lines, 2, 5 + i)
            ax.imshow(data['x'][-1].reshape(self.shape), cmap='gray')
            ax.set_title(method)

        plt.show()
