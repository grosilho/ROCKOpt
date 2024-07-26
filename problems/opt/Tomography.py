from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, vmap

from .RadialBasis import RadialBasis


class Tomography(RadialBasis):
    def __init__(
        self,
        n_theta=50,
        n_v_lines=2,
        n_int_points=100,
        interp_mesh_dx=0.1,
        basis_mesh_dx=0.1,
        free_eps=False,
        free_centers=False,
    ):
        self.n_theta = n_theta
        self.n_v_lines = n_v_lines
        self.n_int_points = n_int_points
        self.N_photons = 1e6

        super().__init__(interp_mesh_dx, basis_mesh_dx, free_eps, free_centers)
        self.name = 'Tomography'  # overwrite the name defined in super().__init__() as "RadialBasis"

    def define_lines(self, dtype_high):
        thetas = jnp.linspace(0.0, jnp.pi, self.n_theta + 1, dtype=dtype_high)[:-1]
        self.ab = []
        self.ab.append(
            jnp.stack([jnp.cos(thetas), jnp.sin(thetas), jnp.cos(thetas + jnp.pi), jnp.sin(thetas + jnp.pi)], axis=1)
        )
        thetas = jnp.linspace(0.0, 2.0 * jnp.pi, self.n_theta + 1, dtype=dtype_high)[:-1]
        for i in range(1, self.n_v_lines):
            x = i / self.n_v_lines
            alpha = jnp.arccos(x)
            self.ab.append(
                jnp.stack(
                    [
                        jnp.cos(thetas - alpha),
                        jnp.sin(thetas - alpha),
                        jnp.cos(thetas + alpha),
                        jnp.sin(thetas + alpha),
                    ],
                    axis=1,
                )
            )

    def define_integrals(self):
        ab = self.ab[0][0]
        norm_ab_max = jnp.linalg.norm(ab[2:] - ab[:2])
        t = jnp.linspace(0.0, 1.0, 2 * self.n_int_points + 1)
        t = t[1::2]
        self.integral_points = []
        self.integral_points.append(
            self.ab[0][:, :2, None] + (self.ab[0][:, 2:, None] - self.ab[0][:, :2, None]) * t[None, None, :]
        )
        for i in range(1, self.n_v_lines):
            ab = self.ab[i][0]
            norm_ab = jnp.linalg.norm(ab[2:] - ab[:2])
            n_int_points = max(int(self.n_int_points * norm_ab / norm_ab_max), 1)
            t = jnp.linspace(0.0, 1.0, 2 * n_int_points + 1)
            t = t[1::2]
            self.integral_points.append(
                self.ab[i][:, :2, None] + (self.ab[i][:, 2:, None] - self.ab[i][:, :2, None]) * t[None, None, :]
            )
        self.dl = [jnp.linalg.norm(int_pts[0, :, 1] - int_pts[0, :, 0]) for int_pts in self.integral_points]

    def integrate_over_lines(self, c, eps, w):
        lines_integrals = [
            jnp.sum(vmap(self.u, in_axes=(0, None, None, None))(points, c, eps, w), axis=1) * dl
            for points, dl in zip(self.integral_points, self.dl)
        ]
        return lines_integrals

    def compute_photons_average(self, lines_integrals):
        photons_average = [self.N_photons * jnp.exp(-li) for li in lines_integrals]
        return photons_average

    def sample_photons(self, photons_average):
        seed = 2000
        key = jax.random.key(seed)
        keys = jax.random.split(key, num=self.n_v_lines)
        # use the Poisson sampling to introduce some noise or the rounded mean to have a more deterministic problem
        # sampled_photons = [jnp.round(photons_average[i]) for i in range(self.n_v_lines)]
        sampled_photons = [jax.random.poisson(keys[i], photons_average[i]) for i in range(self.n_v_lines)]

        return sampled_photons

    def initial_guess(self, dtype):
        self.interpolation_points = self.circle_mesh(self.interp_mesh_dx)
        self.define_lines(dtype.high)
        self.define_integrals()
        self.define_exact_u(dtype)
        self.exact_u_integrals = self.integrate_over_lines(self.u_c, self.u_eps, self.u_w)
        self.exact_photons_average = self.compute_photons_average(self.exact_u_integrals)
        self.sampled_photons = self.sample_photons(self.exact_photons_average)

        self.basis_centers = self.circle_mesh(self.basis_mesh_dx)
        self.n_basis = self.basis_centers.shape[1]
        self.default_eps = 2.0 * jnp.ones(self.n_basis, dtype=dtype.high)
        weights = jnp.ones(self.n_basis, dtype=dtype.high) / self.n_basis
        x0 = weights
        if self.free_eps:
            x0 = jnp.concatenate([x0, self.default_eps])
        if self.free_centers:
            x0 = jnp.concatenate([x0, self.basis_centers.flatten()])

        return x0

    @partial(jit, static_argnums=(0,))
    def f(self, x):
        centers, eps, weights = self.get_c_eps_w(x)
        lines_integrals = self.integrate_over_lines(centers, eps, weights)
        return jnp.mean(
            jnp.array(
                [
                    jnp.mean(self.N_photons * jnp.exp(-li) + samp_pho * li)
                    for li, samp_pho in zip(lines_integrals, self.sampled_photons)
                ]
            )
        )

    def get_metrics(self, x):
        l2_error = jnp.sqrt(
            jnp.mean(
                (
                    self.u(self.interpolation_points, self.u_c, self.u_eps, self.u_w)
                    - self.u(self.interpolation_points, *self.get_c_eps_w(x))
                )
                ** 2
            )
        )
        return {'x': x, 'l2_error': l2_error}

    def get_metrics_keys(self):
        return ['x', 'l2_error']

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
            self.interpolation_points[0, :],
            self.interpolation_points[1, :],
            self.u(self.interpolation_points, self.u_c, self.u_eps, self.u_w),
            cmap=cmap,
            antialiased=False,
            label="Exact",
        )
        for i in range(n_methods):
            ax = fig.add_subplot(2, n_methods + 1, 2 + i, projection='3d')
            method = methods[i]
            centers, eps, weights = self.get_c_eps_w(histories[method]['x'][-1])
            ax.plot_trisurf(
                self.interpolation_points[0, :],
                self.interpolation_points[1, :],
                self.u(self.interpolation_points, centers, eps, weights),
                cmap=cmap,
                antialiased=False,
                label=method,
            )

        ax2 = fig.add_subplot(234)
        ax3 = fig.add_subplot(235)
        ax4 = fig.add_subplot(236)
        self.plot_histories(
            histories=histories,
            keys=["fx", "delta", "l2_error"],
            axes=[ax2, ax3, ax4],
        )

        plt.show()
