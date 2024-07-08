from jax import grad, value_and_grad, jvp, jacfwd, jacrev
from jax import jit
from functools import partial
from utils.print_stuff import plot_options, key_to_tex

import matplotlib.pyplot as plt
import matplotlib.font_manager

matplotlib.rc("font", **{"family": "TeX Gyre DejaVu Math"})
plt.rc("text", usetex=True)


class Problem:
    def __init__(self, name):
        self.name = name

    @partial(jit, static_argnums=(0,))
    def df(self, x):
        # Compute the gradient at x
        df = grad(self.f)(x)
        return df

    @partial(jit, static_argnums=(0,))
    def f_df(self, x):
        # Compute the function value and the gradient at x
        f, df = value_and_grad(self.f)(x)
        return f, df

    @partial(jit, static_argnums=(0,))
    def f_dfv(self, x, v):
        # Compute the function value and the directional derivative dot(grad(f),v), at x
        f, dfv = jvp(self.f, (x,), (v,))
        return f, dfv

    @partial(jit, static_argnums=(0,))
    def ddf(self, x):
        # Compute the full Hessian matrix at x
        ddf = jacfwd(jacrev(self.f))(x)
        return ddf

    @partial(jit, static_argnums=(0,))
    def df_ddfv(self, x, v):
        # Compute the gradient and the Hessian-vector product at x
        # (df(x), ddf(x) * v)
        df, ddfv = jvp(grad(self.f), (x,), (v,))
        return df, ddfv

    def dfv(self, x, v):
        # Compute the directional derivative dot(grad(f),v), at x. Much cheaper than computing the full gradient and then the dot product.
        return self.f_dfv(x, v)[1]

    def ddfv(self, x, v):
        # Compute the Hessian-vector product at x. It is much cheper than computing the full Hessian matrix and then multiply.
        return self.df_ddfv(x, v)[1]

    def get_metrics(self, x):
        return {}

    def get_metrics_keys(self):
        return []

    def plot_histories(self, histories, keys, axes):

        for key, ax in zip(keys, axes):
            for alg_name, history, color, marker in zip(histories.keys(), histories.values(), *plot_options()):
                if key in history:
                    ax.semilogy(history[key], marker=marker, color=color, fillstyle='none', label=alg_name)
            ax.set_title(key_to_tex(key))
            ax.legend()
