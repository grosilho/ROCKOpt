import numpy as np
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


class Rosenbrock(Problem):
    def __init__(self):
        super().__init__('Rosenbrock', 2)

    def initial_guess(self):
        return jnp.array([0.0, -1.0])

    @partial(jit, static_argnums=(0,))
    def f(self, x):
        # Compute the function value at x
        f = (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2
        return f

    def plot(self, history, plot_options):
        import matplotlib
        import matplotlib.pyplot as plt

        plt.set_loglevel(level='warning')

        # Plot countours of the function
        fig = plt.figure()
        ax = fig.add_subplot(111)
        x = jnp.linspace(-2, 2, 100)
        y = jnp.linspace(-2, 2, 100)
        X, Y = jnp.meshgrid(x, y)
        Z = self.f([X, Y])
        levels = 10.0 ** jnp.arange(0, 4, 1)
        ax.contour(X, Y, Z, levels=levels)

        # Plot flows
        for key, val in history.items():
            x = jnp.array([x[0] for x in val["x"]])
            y = jnp.array([x[1] for x in val["x"]])
            plot_opt_str = key.split("|")[0]
            ax.plot(
                x,
                y,
                marker=plot_options[plot_opt_str]["marker"],
                color=plot_options[plot_opt_str]["color"],
                fillstyle='none',
                label=key,
            )

            if plot_options[plot_opt_str]["plot_circles"]:
                [
                    ax.add_patch(
                        matplotlib.patches.Circle(x, r, color=plot_options[key]["color"], fill=True, alpha=0.15)
                    )
                    for x, r in zip(val["x"], val["delta"])
                ]
        ax.legend()
        plt.show()


class MinimalSurface(Problem):
    def __init__(self, M):
        # M is the number of grid points in one direction, thus in total we have M*M grid points
        super().__init__('MinimalSurface', M * M)

        # grid with boundary
        self.sqrt_n = M  # self.n = M * M
        self.x = jnp.linspace(0.0, 1.0, self.sqrt_n + 2)
        self.y = jnp.linspace(0.0, 1.0, self.sqrt_n + 2)
        # mesh size
        self.h = 1.0 / (self.sqrt_n + 1)

    def grid(self, with_boundary):
        # return the frid, with or without the boundary nodes
        if with_boundary:
            return self.x[None, :], self.y[:, None]
        else:
            return self.x[None, 1:-1], self.y[1:-1, None]

    @partial(jit, static_argnums=(0,))
    def g(self, x, y):
        # the boundary condition
        # return y * (1.0 - y) + 0 * x
        freq = 1
        return jnp.cos(freq * 2.0 * jnp.pi * x) * jnp.cos(freq * 2.0 * jnp.pi * y)

    def initial_guess(self):
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
        up = jnp.zeros((self.sqrt_n + 2, self.sqrt_n + 2))
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

    def plot(self, history, plot_options):
        import matplotlib.pyplot as plt
        from matplotlib import cm
        from matplotlib import animation

        X, Y = np.meshgrid(*self.grid(True))

        n_methods = len(history)
        methods = list(history.keys())
        fig, ax = plt.subplots(1, n_methods, subplot_kw={"projection": "3d"})
        if n_methods == 1:
            ax = (ax,)
        cmap = cm.get_cmap("viridis")

        # max_iter = max([len(history[method]["x"]) for method in methods])
        # max_iter = min(max_iter, 50)
        anim_iter = 50

        def plot_u(n):
            for i in range(n_methods):
                x = history[methods[i]]["x"]
                ind = np.round(n / (anim_iter - 1) * (len(x) - 1)).astype(int)
                ax[i].cla()
                up = self.add_g(x[ind].reshape(self.sqrt_n, self.sqrt_n))
                ax[i].plot_surface(X, Y, up, cmap=cmap, linewidth=0, antialiased=False)
                ax[i].set(xlim=[0.0, 1.0], ylim=[0.0, 1.0], zlim=[-1.0, 1.0], xlabel='x', ylabel='y', zlabel='f')
                ax[i].set_title(f"{methods[i].split("|")[0]}\n{ind+1}/{len(x)}")

        plot_u(0)

        def update(frame):
            plot_u(frame)
            return (fig,)

        duration = 5.0
        fps = anim_iter / duration
        anim = animation.FuncAnimation(fig=fig, func=update, frames=anim_iter, interval=1000 / fps, repeat=False)
        plt.show()
