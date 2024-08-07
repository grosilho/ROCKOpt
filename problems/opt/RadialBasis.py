import jax
import jax.numpy as jnp
from jax import jit, lax
from functools import partial

from .Problem import Problem


class RadialBasis(Problem):
    def __init__(self, interp_mesh_dx=0.1, basis_mesh_dx=0.1, free_eps=False, free_centers=False):
        self.interp_mesh_dx = interp_mesh_dx
        self.basis_mesh_dx = basis_mesh_dx
        self.free_eps = free_eps
        self.free_centers = free_centers
        super().__init__('RadialBasis')

    def circle_mesh(self, dx, dtype):
        n_r = jnp.round(1.0 / dx).astype(jnp.int32)
        rs = jnp.linspace(0.0, 1.0, n_r + 1)[:-1]
        dr = rs[1] - rs[0]
        mesh = jnp.array([0.0, 0.0]).reshape(2, 1)
        for i in range(1, n_r):
            r = rs[i]
            dtheta = jnp.arccos(1.0 - (dr / (jnp.sqrt(2.0) * r)) ** 2)
            n_theta = jnp.ceil(2.0 * jnp.pi / dtheta).astype(jnp.int32)
            thetas = jnp.linspace(0.0, 2.0 * jnp.pi, n_theta + 1)[:-1]
            mesh = jnp.concatenate(
                [
                    mesh,
                    jnp.stack([r * jnp.cos(thetas), r * jnp.sin(thetas)], axis=0),
                ],
                axis=1,
            )

        mp_mesh = dict()
        mp_mesh[dtype.high.dtype] = mesh
        if dtype.low is not None:
            mp_mesh[dtype.low.dtype] = lax.convert_element_type(mesh, self.dtype.low)

        return mp_mesh

    def initial_guess(self, dtype):
        self.dtype = dtype
        self.define_exact_u(dtype)
        self.interpolation_points = self.circle_mesh(self.interp_mesh_dx, dtype)
        self.default_basis_centers = self.circle_mesh(self.basis_mesh_dx, dtype)
        self.n_basis = self.default_basis_centers[dtype.high.dtype].shape[1]
        self.default_eps = jnp.ones(self.n_basis, dtype=dtype.high)
        weights = jnp.ones(self.n_basis, dtype=dtype.high) / self.n_basis
        x0 = weights
        if self.free_eps:
            x0 = jnp.concatenate([x0, self.default_eps])
        if self.free_centers:
            x0 = jnp.concatenate([x0, self.default_basis_centers[dtype.high.dtype].flatten()])
        self.default_eps = {dtype.high.dtype: self.default_eps}
        if dtype.low is not None:
            self.default_eps[dtype.low.dtype] = lax.convert_element_type(self.default_eps[dtype.high.dtype], dtype.low)

        return x0

    def define_exact_u(self, dtype):
        n_basis = 3
        min_eps = 1.0
        max_eps = 10.0
        min_w = 0.0
        max_w = 5.0

        seed = 1989
        key = jax.random.key(seed)
        keys = jax.random.split(key, num=4)
        rs = jax.random.uniform(keys[0], shape=(n_basis,), dtype=dtype.high)
        thetas = 2.0 * jnp.pi * jax.random.uniform(keys[1], shape=(n_basis,), dtype=dtype.high)
        self.u_c = {dtype.high.dtype: jnp.stack([rs * jnp.cos(thetas), rs * jnp.sin(thetas)], axis=0)}
        self.u_eps = {
            dtype.high.dtype: min_eps
            + (max_eps - min_eps) * jax.random.uniform(keys[2], shape=(n_basis,), dtype=dtype.high)
        }
        self.u_w = {
            dtype.high.dtype: min_w + (max_w - min_w) * jax.random.uniform(keys[3], shape=(n_basis,), dtype=dtype.high)
        }
        if dtype.low is not None:
            self.u_c[dtype.low.dtype] = lax.convert_element_type(self.u_c[dtype.high.dtype], dtype.low)
            self.u_eps[dtype.low.dtype] = lax.convert_element_type(self.u_eps[dtype.high.dtype], dtype.low)
            self.u_w[dtype.low.dtype] = lax.convert_element_type(self.u_w[dtype.high.dtype], dtype.low)

    def phi(self, x, c, eps):
        diff = x[:, :, None] - c[:, None, :]
        norm_sq_diff = jnp.sum(diff**2, axis=0)
        return jnp.exp(-eps[None, :] ** 2 * norm_sq_diff)

    def u(self, x, c, eps, w):
        return jnp.matmul(self.phi(x, c, eps), w)

    @partial(jit, static_argnums=(0,))
    def f(self, x):
        centers, eps, weights = self.get_c_eps_w(x)
        diff = self.u(
            self.interpolation_points[x.dtype], self.u_c[x.dtype], self.u_eps[x.dtype], self.u_w[x.dtype]
        ) - self.u(self.interpolation_points[x.dtype], centers, eps, weights)
        return jnp.mean(diff**2)

    def get_c_eps_w(self, x):
        weights = x[: self.n_basis]

        if not self.free_eps:
            eps = self.default_eps[x.dtype]
        else:
            eps = x[self.n_basis : 2 * self.n_basis]

        if self.free_centers:
            centers = x[self.n_basis + int(self.free_eps) * self.n_basis :]
            centers = centers.reshape(2, self.n_basis)
        else:
            centers = self.default_basis_centers[x.dtype]

        return centers, eps, weights

    def get_metrics(self, x):
        return {'x': x}

    def get_metrics_keys(self):
        return ['x']

    def plot(self, histories):
        import matplotlib
        import matplotlib.pyplot as plt

        n_methods = len(histories)
        assert n_methods <= 3, "Only <=3 methods can be compared."

        methods = list(histories.keys())

        plt.set_loglevel(level='warning')
        figsize = (5 * (n_methods + 1), 10)
        fig = plt.figure(figsize=figsize)

        cmap = matplotlib.colormaps['viridis']

        ax1 = fig.add_subplot(2, n_methods + 1, 1, projection='3d')
        ax1.plot_trisurf(
            self.interpolation_points[self.dtype.high.dtype][0, :],
            self.interpolation_points[self.dtype.high.dtype][1, :],
            self.u(
                self.interpolation_points[self.dtype.high.dtype],
                self.u_c[self.dtype.high.dtype],
                self.u_eps[self.dtype.high.dtype],
                self.u_w[self.dtype.high.dtype],
            ),
            cmap=cmap,
            antialiased=False,
        )
        for i in range(n_methods):
            ax = fig.add_subplot(2, n_methods + 1, 2 + i, projection='3d')
            method = methods[i]
            centers, eps, weights = self.get_c_eps_w(histories[method]['x'][-1])
            ax.plot_trisurf(
                self.interpolation_points[self.dtype.high.dtype][0, :],
                self.interpolation_points[self.dtype.high.dtype][1, :],
                self.u(self.interpolation_points[self.dtype.high.dtype], centers, eps, weights),
                cmap=cmap,
                antialiased=False,
            )

        ax2 = fig.add_subplot(223)
        ax3 = fig.add_subplot(224)
        self.plot_histories(
            histories=histories,
            keys=["fx", "delta"],
            axes=[ax2, ax3],
        )

        plt.show()
