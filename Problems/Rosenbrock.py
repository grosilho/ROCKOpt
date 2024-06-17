import jax.numpy as jnp
from jax import jit
from functools import partial

from print_stuff import plot_options
from .Problem import Problem


class Rosenbrock(Problem):
    def __init__(self, dim=2):
        self.dim = dim
        assert self.dim == 2 or self.dim == 3, "Rosenbrock only supports 2 or 3 dimensions"
        super().__init__('Rosenbrock', dim)

    def initial_guess(self):
        if self.dim == 2:
            return jnp.array([-1.0, -1.0])
        elif self.dim == 3:
            return jnp.array([-1.0, -1.0, -1.0])

    @partial(jit, static_argnums=(0,))
    def f(self, x):
        # Compute the function value at x
        if self.dim == 2:
            f = (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2
        elif self.dim == 3:
            f = 100 * ((x[2] - x[1] ** 2) ** 2 + (x[1] - x[0] ** 2) ** 2) + (1.0 - x[0]) ** 2 + (1.0 - x[1]) ** 2
        return f

    def plot(self, history):
        """
        Plot the solution path, the function value and the trust region radius for each iteration and each method

        Args:
            history: dictionary of dictionaries with the following structure:
            {
                "method1": {
                    "x": [x_1, x_2, ..., x_n],
                    "fx": [f(x_1), f(x_2), ..., f(x_n)],
                    "delta": [delta_1, delta_2, ..., delta_n]
                },
                "method2": {
                    "x": [x_1, x_2, ..., x_n],
                    "fx": [f(x_1), f(x_2), ..., f(x_n)],
                    "delta": [delta_1, delta_2, ..., delta_n]
                },
                ...
            }
        """
        import matplotlib.pyplot as plt

        plt.set_loglevel(level='warning')
        figsize = (15, 5)
        fig = plt.figure(figsize=figsize)

        if self.dim == 2:
            # Plot countours of the function
            ax1 = fig.add_subplot(131)
            x = jnp.linspace(-2, 2, 100)
            y = jnp.linspace(-2, 2, 100)
            X, Y = jnp.meshgrid(x, y)
            Z = self.f([X, Y])
            levels = 10.0 ** jnp.arange(0, 4, 1)
            ax1.contour(X, Y, Z, levels=levels)

            # plot the paths
            for key, val, color, marker in zip(history.keys(), history.values(), *plot_options()):
                x = jnp.array([x[0] for x in val["x"]])
                y = jnp.array([x[1] for x in val["x"]])
                ax1.plot(x, y, marker=marker, color=color, fillstyle='none', label=key)

            ax1.legend()

        elif self.dim == 3:

            ax1 = fig.add_subplot(131, projection='3d')
            # plot the paths
            for key, val, color, marker in zip(history.keys(), history.values(), *plot_options()):
                x = jnp.array([x[0] for x in val["x"]])
                y = jnp.array([x[1] for x in val["x"]])
                z = jnp.array([x[2] for x in val["x"]])
                ax1.plot(x, y, z, marker=marker, color=color, fillstyle='none', label=key)

            ax1.legend()

        ax1.set_title("Paths")

        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)
        self.plot_fx_and_delta(ax2, ax3, history)
        plt.show()
