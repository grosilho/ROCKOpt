import jax.numpy as jnp
from jax import jit
from functools import partial

from .Problem import Problem


class MinimalSurface(Problem):
    def __init__(self, M):
        # M is the number of grid points in one direction, thus in total we have M*M grid points
        super().__init__('MinimalSurface')

        # grid with boundary
        self.sqrt_n = M  # self.n = M * M
        # mesh size
        self.h = 1.0 / (self.sqrt_n + 1)

    def init_grid(self, dtype):
        self.x = jnp.linspace(0.0, 1.0, self.sqrt_n + 2, dtype=dtype)
        self.y = jnp.linspace(0.0, 1.0, self.sqrt_n + 2, dtype=dtype)

    def grid(self, with_boundary):
        # return the frid, with or without the boundary nodes
        if with_boundary:
            return self.x[None, :], self.y[:, None]
        else:
            return self.x[None, 1:-1], self.y[1:-1, None]

    @partial(jit, static_argnums=(0,))
    def g(self, x, y):
        # the boundary condition
        freq = 1
        r = freq * 2.0 * jnp.pi
        return jnp.cos(r * x) * jnp.cos(r * y)

    def initial_guess(self, dtype):
        self.init_grid(dtype.high)
        return self.g(*self.grid(False)).flatten()

    @partial(jit, static_argnums=(0,))
    def set_g(self, up):
        # impose the boundary condition
        up = up.at[0, :].set(self.g(self.x, 0.0))
        up = up.at[-1, :].set(self.g(self.x, 1.0))
        up = up.at[:, 0].set(self.g(0.0, self.y))
        up = up.at[:, -1].set(self.g(1.0, self.y))
        return up

    @partial(jit, static_argnums=(0,))
    def add_g(self, u):
        up = jnp.zeros((self.sqrt_n + 2, self.sqrt_n + 2), dtype=u.dtype)
        up = up.at[1:-1, 1:-1].set(u)
        up = self.set_g(up)
        return up

    @partial(jit, static_argnums=(0,))
    def grad_u(self, u):
        up = self.add_g(u)
        dudx = (up[:-1, 1:] + up[1:, 1:] - up[:-1, :-1] - up[1:, :-1]) / (2 * self.h)
        dudy = (up[1:, :-1] + up[1:, 1:] - up[:-1, :-1] - up[:-1, 1:]) / (2 * self.h)
        return dudx, dudy

    @partial(jit, static_argnums=(0,))
    def f(self, u):
        ur = u.reshape((self.sqrt_n, self.sqrt_n))
        dudx, dudy = self.grad_u(ur)
        f = self.h**2 * jnp.sum(jnp.sqrt((dudx**2 + dudy**2) + 1.0))
        return f

    def get_metrics(self, x):
        return {"x": x}

    def get_metrics_keys(self):
        return ["x"]

    def plot(self, histories):
        import matplotlib
        import matplotlib.pyplot as plt
        from matplotlib import animation

        n_methods = len(histories)
        assert n_methods <= 2, "Only 1 or 2 methods can be compared."

        methods = list(histories.keys())

        figsize = (15, 10)
        fig = plt.figure(figsize=figsize)

        ax = [[None, None], [None, None]]
        ax[0][0] = fig.add_subplot(221, projection='3d')
        ax[0][1] = fig.add_subplot(222, projection='3d')
        ax[1][0] = fig.add_subplot(223)
        ax[1][1] = fig.add_subplot(224)

        cmap = matplotlib.colormaps['viridis']

        anim_iter = 25

        x, y = self.grid(True)
        X, Y = jnp.meshgrid(x.flatten(), y.flatten())

        def plot_u(n):
            for i in range(n_methods):
                x = histories[methods[i]]["x"]
                ind = jnp.round(n / (anim_iter - 1) * (len(x) - 1)).astype(int)
                ax[0][i].cla()
                up = self.add_g(x[ind].reshape(self.sqrt_n, self.sqrt_n))
                ax[0][i].plot_surface(X, Y, up, cmap=cmap, linewidth=0, antialiased=False)
                ax[0][i].set(xlim=[0.0, 1.0], ylim=[0.0, 1.0], zlim=[-1.0, 1.0], xlabel='x', ylabel='y', zlabel='f')
                ax[0][i].set_title(f"{methods[i].split("|")[0]}\n{ind+1}/{len(x)}")

        self.plot_histories(histories=histories, keys=["fx", "delta"], axes=[ax[1][0], ax[1][1]])

        plot_u(0)

        def update(frame):
            plot_u(frame)
            return (fig,)

        duration = 5.0
        fps = anim_iter / duration
        anim = animation.FuncAnimation(fig=fig, func=update, frames=anim_iter, interval=1000 / fps, repeat=False)

        plt.show()
