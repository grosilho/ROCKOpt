import jax
import jax.numpy as jnp
import os

jax.config.update('jax_platform_name', 'cpu')  # Use CPU as default device
# os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=12'  # Number of cores
jax.config.update("jax_enable_x64", True)  # Enable 64-bit precision but it is much slower on GPU

# print(f"Jax Available CPU Devices: {jax.devices("cpu")}")
# print(f"Jax Available GPU Devices: {jax.devices("gpu")}")
print(f"Jax Default Backend: {jax.default_backend()}")
print(f"Jax Default Device: {jnp.ones(3).devices()}")

import logging
from MinimizationAlgorithms import (
    TrustRegion,
    StabilizedTrustRegion,
    StabilizedGradientFlow,
    StabilizedNewtonFlow,
    SplitStabilizedNewtonFlow,
    ExactGradientFlow,
    ExactNewtonFlow,
)
from Problems import Rosenbrock, MinimalSurface, Madelon
import print_stuff

logging.basicConfig(level=logging.INFO)

# Define the function to minimize
# Fun = MinimalSurface(M=128)
# Fun = Rosenbrock(dim=3)
Fun = Madelon()
# Define the initial guess
x0 = Fun.initial_guess()

# Set a list of minimization algorithms to run and compare
min_algs = [
    "TrustRegion",
    # "StabilizedTrustRegion",
    "StabilizedGradientFlow",
    # "StabilizedNewtonFlow",
    # "SplitStabilizedNewtonFlow",
    # "ExactGradientFlow",
    # "ExactNewtonFlow",
]

common_options = options = {"n": Fun.n, "max_iter": 1e2, "rtol": 1e-6, "atol": 0.0}

specific_options = dict()
specific_options["TrustRegion"] = {
    "delta_max": 1.0,
    "eta": 1e-4,
    "loc_prob_sol": "dog_leg",
    "method": "direct",
    "iter_solver_tol": 1e-5,
    "iter_solver_maxiter": 100,
}
specific_options["StabilizedTrustRegion"] = {
    "delta_max": 1.0,
    "eta": 1e-4,
    "method": "RKC1",
    "damping": 1.0,
    "safe_add": 2,
    "dt": 1e-6,
    "max_steps": 1000,
    "p_conv_tol": 1e-2,
    "rho_freq": 1,
}
specific_options["StabilizedGradientFlow"] = {
    "delta_max": 1e-4,
    "method": "RKC1",
    "damping": 1.0,
    "safe_add": 2,
    "rho_freq": 1,
    "record_stages": False,
}
specific_options["StabilizedNewtonFlow"] = {
    "delta_max": 1e-6,
    "method": "RKC1",
    "damping": 1.0,
    "safe_add": 2,
    "eps": 1.0,
    "rho_freq": 1,
    "record_stages": False,
}
specific_options["SplitStabilizedNewtonFlow"] = {
    "delta_max": 1e-5,
    "method": "RKC1",
    "damping": 1.0,
    "safe_add": 2,
    "eps": 0.1,
    "rho_freq": 1,
}
specific_options["ExactGradientFlow"] = {"delta_max": 1e-2}
specific_options["ExactNewtonFlow"] = {"delta_max": 1.0}

history = dict()
stats = dict()
# Define the minimization algorithm and solve the problem
for name in min_algs:
    min_algo = eval(name)(**common_options, **specific_options[name])
    description = min_algo.description
    history[description], stats[description] = min_algo.solve(Fun, x0)


print_stuff.print_table(stats)

Fun.plot(history, print_stuff.plot_options())
