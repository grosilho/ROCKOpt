from Functions_old import Rosenbrock
from Functions import Rosenbrock as JAX_Rosenbrock
from Functions import MinimalSurface as JAX_MinimalSurface
import numpy as np
import jax.numpy as jnp
from jax import random, jit
from jax.test_util import check_grads
import matplotlib.pyplot as plt
from matplotlib import cm


def test_Rosenbrock():
    rb = Rosenbrock()
    jax_rb = JAX_Rosenbrock()

    x = jnp.array([1.0, 1.0])

    assert np.allclose(rb.f(x), jax_rb.f(x), rtol=1e-5)
    assert np.allclose(rb.df(x), jax_rb.df(x), rtol=1e-5)
    assert np.allclose(rb.ddf(x), jax_rb.ddf(x), rtol=1e-5)

    f, df = jax_rb.f_df(x)
    assert np.allclose(rb.f(x), f, rtol=1e-5)
    assert np.allclose(rb.df(x), df, rtol=1e-5)

    key = random.key(42)
    v = random.normal(key, shape=(2,))
    f, dfv = jax_rb.f_dfv(x, v)
    assert np.allclose(rb.f(x), f, rtol=1e-5)
    assert np.allclose(rb.df(x) @ v, dfv, rtol=1e-5)

    df, ddfv = jax_rb.df_ddfv(x, v)
    assert np.allclose(rb.df(x), df, rtol=1e-5)
    assert np.allclose(rb.ddf(x) @ v, ddfv, rtol=1e-5)


def test_MinimalSurface():

    n = 128
    ms = JAX_MinimalSurface(n)

    x, y = ms.grid(with_boundary=False)
    u = ms.g(x, y).ravel()
    v = ms.g(x, y).ravel()
    v = v / jnp.linalg.norm(v)

    X, Y = np.meshgrid(x, y)
    ur = u.reshape(n, n)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(X, Y, ur, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.set(xlim=[0.0, 1.0], ylim=[0.0, 1.0], zlim=[-1.0, 1.0], xlabel='x', ylabel='y')
    plt.show()
    exit()

    dx = 1e-1 / 8
    f = ms.f(u)
    fj, dfv = ms.f_dfv(u, v)
    dfv_num = (ms.f(u + dx * v) - ms.f(u)) / dx
    err = abs(dfv - dfv_num) / abs(dfv_num)

    # print(f"f: {f}")
    # print(f"fj: {fj}")
    # print(f"dfv_num: {dfv_num}")
    # print(f"dfv: {dfv}")
    # print(f"err = {err}")

    dx = 1e-2
    fj, df = ms.f_df(u)
    df_num = jnp.zeros(n * n)
    v = jnp.zeros(n * n)
    for i in range(n * n):
        vi = v.at[i].set(1.0)
        df_num = df_num.at[i].set((ms.f(u + dx * vi) - ms.f(u - dx * vi)) / dx / 2.0)
    err = np.linalg.norm(df - df_num) / np.linalg.norm(df)

    print(f"f: {f}")
    print(f"fj: {fj}")
    print(f"df_num: {np.linalg.norm(dfv_num)}")
    print(f"df: {np.linalg.norm(dfv)}")
    print(f"err = {err}")

    dfr = df.reshape(n, n)
    dfr_num = df_num.reshape(n, n)

    ur = u.reshape(n, n)
    dudx, dudy = ms.grad_u(ur)
    r = jnp.sqrt((dudx**2 + dudy**2) + 1.0)[:-1, :-1]

    X, Y = np.meshgrid(x, y)
    # fig, ax = plt.subplots(2, 2, subplot_kw={"projection": "3d"})
    # ax[0, 0].plot_surface(X, Y, ur, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    # ax[0, 1].plot_surface(X, Y, dfr, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    # ax[1, 1].plot_surface(X, Y, dfr_num, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    # ax[1, 0].plot_surface(X, Y, np.abs(dfr - dfr_num), cmap=cm.coolwarm, linewidth=0, antialiased=False)
    # plt.show()


if __name__ == '__main__':
    # test_Rosenbrock()
    test_MinimalSurface()
