import jax.numpy as jnp
from jax import jit
from functools import partial

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

    def plot(self, history, plot_options):
        import matplotlib
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

            # Plot flows
            for key, val in history.items():
                x = jnp.array([x[0] for x in val["x"]])
                y = jnp.array([x[1] for x in val["x"]])
                plot_opt_str = key.split("|")[0]
                ax1.plot(
                    x,
                    y,
                    marker=plot_options[plot_opt_str]["marker"],
                    color=plot_options[plot_opt_str]["color"],
                    fillstyle='none',
                    label=key,
                )

                if plot_options[plot_opt_str]["plot_circles"]:
                    [
                        ax1.add_patch(
                            matplotlib.patches.Circle(x, r, color=plot_options[key]["color"], fill=True, alpha=0.15)
                        )
                        for x, r in zip(val["x"], val["delta"])
                    ]
            ax1.legend()

        elif self.dim == 3:

            ax1 = fig.add_subplot(131, projection='3d')

            # Plot flows
            for key, val in history.items():
                x = jnp.array([x[0] for x in val["x"]])
                y = jnp.array([x[1] for x in val["x"]])
                z = jnp.array([x[2] for x in val["x"]])
                plot_opt_str = key.split("|")[0]
                ax1.plot(
                    x,
                    y,
                    z,
                    marker=plot_options[plot_opt_str]["marker"],
                    color=plot_options[plot_opt_str]["color"],
                    fillstyle='none',
                    label=key,
                )

                if plot_options[plot_opt_str]["plot_circles"]:
                    [
                        ax1.add_patch(
                            matplotlib.patches.Circle(x, r, color=plot_options[key]["color"], fill=True, alpha=0.15)
                        )
                        for x, r in zip(val["x"], val["delta"])
                    ]
            ax1.legend()

        ax1.set_title("Paths")
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)
        self.plot_fx_and_delta(ax2, ax3, history, plot_options)
        plt.show()
