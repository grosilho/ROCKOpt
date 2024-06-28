import jax
import jax.numpy as jnp

import logging
from Solvers.Opt import (
    TrustRegion,
    StabilizedTrustRegion,
    StabilizedGradientFlow,
    StabilizedNewtonFlow,
    SplitStabilizedNewtonFlow,
    ExactGradientFlow,
    ExactNewtonFlow,
)
from Problems.Opt import Madelon
import print_stuff

gpu = True
jit = True
float64 = False
profile = False

if not gpu:
    jax.config.update('jax_platform_name', 'cpu')  # Use CPU as default device
if float64:
    jax.config.update("jax_enable_x64", True)  # Enable 64-bit precision but it is much slower on GPU
# os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=12'  # Number of cores

# print(f"Jax Available CPU Devices: {jax.devices("cpu")}")
# print(f"Jax Available GPU Devices: {jax.devices("gpu")}")
print(f"Jax Default Backend: {jax.default_backend()}")
print(f"Jax Default Device: {jnp.ones(3).devices()}")

logging.basicConfig(level=logging.INFO)

# Define the function to minimize
Fun = Madelon()
x0 = Fun.initial_guess()

# Set a list of minimization algorithms to run and compare
min_algs = [
    # "TrustRegion",
    "StabilizedTrustRegion",
    # "StabilizedGradientFlow",
    # "StabilizedNewtonFlow",
    # "SplitStabilizedNewtonFlow",
    # "ExactGradientFlow",
    # "ExactNewtonFlow",
]

# all methods use the same stopping criteria, and these are its parameters
common_options = {"max_iter": 1e3, "rtol": 5e-3, "atol": 0.0}

specific_options = dict()
specific_options["TrustRegion"] = {
    "delta_max": 2.0,
    "eta": 1e-4,
    "loc_prob_sol": "dog_leg",
    "method": "iterative",
    "iter_solver_tol": 1e-4,
    "iter_solver_maxiter": 100,
}
specific_options["StabilizedTrustRegion"] = {
    "delta_max": 2.0,
    "eta": 1e-4,
    "method": "RKC1",
    "damping": 10.0,
    "safe_add": 2,
    "dt": 0.1,
    "max_steps": 100,
    "p_conv_tol": 1e-3,
    "rho_freq": 1,
}
specific_options["StabilizedGradientFlow"] = {
    "delta_max": 20.0,
    "method": "RKC1",
    "damping": 1.0,
    "safe_add": 1,
    "rho_freq": 5,
    "record_stages": False,
}
specific_options["StabilizedNewtonFlow"] = {
    "delta_max": 0.1,
    "method": "RKC1",
    "damping": 10.0,
    "safe_add": 2,
    "eps": 1e-1,
    "rho_freq": 10,
    "record_stages": False,
}
specific_options["SplitStabilizedNewtonFlow"] = {
    "delta_max": 1.0,
    "method": "RKC1",
    "damping": 40.0,
    "safe_add": 2,
    "eps": 1.0,
    "rho_freq": 10,
}
specific_options["ExactGradientFlow"] = {"delta_max": 1e-3}
specific_options["ExactNewtonFlow"] = {"delta_max": 1.0}


# Initialize the minimization algorithms and perform one iteration
# We perform one iteration and discrad the results to compile the code
# This is done to avoid the compilation time in the actual simulation
# and do not measure compilatio time in the performance evaluation
min_algos = dict()
log_level = logging.getLogger().getEffectiveLevel()
for name in min_algs:
    min_algos[name] = eval(name)(**common_options, **specific_options[name])
    min_algos[name].logger.setLevel(logging.ERROR)
    min_algos[name].solve(Fun, x0, max_iter=1)
    min_algos[name].logger.setLevel(log_level)

history = dict()
stats = dict()
# Solve the problem
with jax.disable_jit(not jit):
    if profile:
        jax.profiler.start_trace("/tmp/jax-trace", create_perfetto_link=True)
    for name in min_algs:
        description = min_algos[name].description
        history[description], stats[description] = min_algos[name].solve(Fun, x0)
    if profile:
        jax.profiler.stop_trace()


print_stuff.print_table(stats)

Fun.plot(history)
