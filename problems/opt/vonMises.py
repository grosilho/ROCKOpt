import jax
import jax.numpy as jnp
from jax import jit, lax
from functools import partial
from scipy.stats import vonmises_fisher

from .Problem import Problem


class vonMises(Problem):
    def __init__(self, kappa, dim, n_samples):
        self.kappa = kappa
        self.dim = dim
        self.n_samples = n_samples
        self.mu_seed = 2024
        self.init_guess_seed = 1989
        self.samples_seed = 2000

        self.lmbda = 1000.0

        assert kappa > 0.0, "kappa must be strictly positive"
        assert type(dim) is int, "dim must be an integer"
        assert dim > 1, "dim must be greater than 1"
        assert type(n_samples) is int, "n_samples must be an integer"

        super().__init__('vonMises')

    def initial_guess(self, dtype):
        self.init_samples(dtype)
        key = jax.random.key(self.init_guess_seed)
        key, subkey = jax.random.split(key)
        mu = jax.random.uniform(key, shape=(self.dim,), dtype=dtype.high)
        mu = mu / jnp.linalg.norm(mu)
        kappa_0 = jax.random.uniform(subkey, dtype=dtype.high)
        return jnp.vstack([mu.reshape((self.dim, 1)), kappa_0]).reshape(-1)

    def init_samples(self, dtype):
        key = jax.random.key(self.mu_seed)
        self.mu = jax.random.uniform(key, shape=(self.dim,), dtype=dtype.high)
        self.mu = self.mu / jnp.linalg.norm(self.mu)
        samples = vonmises_fisher(self.mu, self.kappa, self.samples_seed).rvs(self.n_samples)
        self.suff_stat = dict()
        self.suff_stat[dtype.high.dtype] = jnp.mean(samples, axis=0)
        if dtype.low is not None:
            self.suff_stat[dtype.low.dtype] = lax.convert_element_type(
                self.suff_stat[dtype.high.dtype], dtype.low.dtype
            )

    @partial(jit, static_argnums=(0,))
    def f(self, x):
        d = self.dim
        mu = x[:-1]
        kappa = x[-1].astype(float)
        if d == 2:
            minus_log_Cd = jnp.log(2.0 * jnp.pi * jax.scipy.special.i0(kappa))
        elif d == 3:
            minus_log_Cd = jnp.log((2.0 * jnp.pi * (jnp.exp(kappa) - jnp.exp(-kappa))) / kappa)
        elif d == 4:
            minus_log_Cd = jnp.log(((2.0 * jnp.pi) ** 2 * jax.scipy.special.i1(kappa)) / kappa)
        else:
            minus_log_Cd = (
                jnp.log(2.0 * jnp.pi) * (d / 2.0)
                + jnp.log(self.mod_Bessel_first_kind(kappa))
                - jnp.log(kappa) * (d / 2.0 - 1.0)
            )

        return minus_log_Cd - kappa * jnp.dot(mu, self.suff_stat[mu.dtype]) + self.lmbda * (jnp.dot(mu, mu) - 1.0) ** 2

    @partial(jit, static_argnums=(0,))
    def mod_Bessel_first_kind(self, x):

        alpha = self.dim / 2.0 - 1.0
        N1 = 50 * self.dim
        N2 = 1000
        large = 10.0

        theta = jnp.linspace(0.0, jnp.pi, N1 + 1)[:-1]
        t = jnp.linspace(0.0, large, N2 + 1)[:-1]
        dtheta = jnp.pi / N1
        dt = large / N2

        I1 = dtheta * jnp.sum(jnp.exp(x * jnp.cos(theta)) * jnp.cos(alpha * theta))
        I2 = dt * jnp.sum(jnp.exp(-x * jnp.cosh(t) - alpha * t))
        result = (I1 - jnp.sin(alpha * jnp.pi) * I2) / jnp.pi

        return result

    def get_metrics(self, x):
        return {
            "mu_error": jnp.linalg.norm(x[:-1] - self.mu),
            "kappa_error": jnp.abs(x[-1] - self.kappa),
            "norm_mu": jnp.linalg.norm(x[:-1]),
            "angle": jnp.acos(jnp.dot(x[:-1], self.mu) / jnp.linalg.norm(x[:-1])),
            "kappa": x[-1],
        }

    def get_metrics_keys(self):
        return ["mu_error", "kappa_error", "norm_mu", "angle", "kappa"]

    def plot(self, histories):
        import matplotlib.pyplot as plt

        plt.set_loglevel(level='warning')
        figsize = (15, 5)
        fig = plt.figure(figsize=figsize)

        axes = fig.subplots(nrows=3, ncols=3)
        self.plot_histories(
            histories=histories,
            keys=["fx", "delta", "mu_error", "kappa_error", "norm_mu", "angle", "kappa"],
            axes=axes.ravel(),
        )

        plt.show()


# from utils.common import MP_dtype

# mp_dtype = MP_dtype(jnp.float32, jnp.float16)
# problem = vonMises(10.0, 6, 10)
# x = problem.initial_guess(mp_dtype)
# fx = problem.f(x)
# dfx = problem.df(x)

# print(f"x: {x}")
# print(f"fx: {fx}")
# print(f"dfx: {dfx}")
