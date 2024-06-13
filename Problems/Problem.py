import jax.numpy as jnp
from jax import grad, value_and_grad, jvp, jacfwd, jacrev
from jax import jit
from functools import partial


class Problem:
    def __init__(self, name, n):
        self.name = name
        self.n = n

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

    def plot_fx_and_delta(self, ax1, ax2, history, plot_options):
        for key, val in history.items():
            fx = jnp.array(val["fx"])
            delta = jnp.array(val["delta"])
            iter = jnp.arange(0, len(fx))
            plot_opt_str = key.split("|")[0]
            ax1.semilogy(
                iter,
                fx,
                marker=plot_options[plot_opt_str]["marker"],
                color=plot_options[plot_opt_str]["color"],
                fillstyle='none',
                label=key,
            )
            ax2.plot(
                iter,
                delta,
                marker=plot_options[plot_opt_str]["marker"],
                color=plot_options[plot_opt_str]["color"],
                fillstyle='none',
                label=key,
            )
        ax1.set_title("Function value")
        ax2.set_title("Trust region radius")
        ax1.legend()
        ax2.legend()
