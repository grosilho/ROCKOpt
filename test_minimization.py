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
from Functions import Rosenbrock, MinimalSurface
import print_stuff

logging.basicConfig(level=logging.WARNING)

# Define the function to minimize
Fun = MinimalSurface(M=64)
# Fun = Rosenbrock()
# Define the initial guess
x0 = Fun.initial_guess()

# Set a list of minimization algorithms to run and compare
min_algs = [
    "TrustRegion",
    "StabilizedTrustRegion",
    # "StabilizedGradientFlow",
    # "StabilizedNewtonFlow",
    # "SplitStabilizedNewtonFlow",
    # "ExactGradientFlow",
    # "ExactNewtonFlow",
]

common_options = options = {"n": Fun.n, "max_iter": 1e6, "tol": 1e-4}

specific_options = dict()
specific_options["TrustRegion"] = {
    "delta_max": 1.0,
    "eta": 1e-4,
    "loc_prob_sol": "dog_leg",
    "method": "iterative" if isinstance(Fun, MinimalSurface) else "direct",
    "iter_solver_tol": 1e-5,
    "iter_solver_maxiter": 100,
}
specific_options["StabilizedTrustRegion"] = {
    "delta_max": 1.0,
    "eta": 1e-4,
    "method": "RKC1",
    "damping": 0.05,
    "safe_add": 1,
    "dt": 1.0,
    "max_steps": 20,
    "p_conv_tol": 1e-2,
    "rho_freq": 1,
}
specific_options["StabilizedGradientFlow"] = {
    "delta_max": 0.5,
    "method": "RKC1",
    "damping": 5.0,
    "safe_add": 1,
    "rho_freq": 1,
    "record_stages": False,
}
specific_options["StabilizedNewtonFlow"] = {
    "delta_max": 0.5,
    "method": "RKC1",
    "damping": 5.0,
    "safe_add": 1,
    "eps": 0.01,
    "rho_freq": 1,
    "record_stages": False,
}
specific_options["SplitStabilizedNewtonFlow"] = {
    "delta_max": 0.5,
    "method": "RKC1",
    "damping": 15.0,
    "safe_add": 2,
    "eps": 0.025,
    "rho_freq": 1,
}
specific_options["ExactGradientFlow"] = {"delta_max": 1e-2}
specific_options["ExactNewtonFlow"] = {"delta_max": 0.1}

history = dict()
stats = dict()
# Define the minimization algorithm and solve the problem
for name in min_algs:
    min_algo = eval(name)(**common_options, **specific_options[name])
    description = min_algo.description
    history[description], stats[description] = min_algo.solve(Fun, x0)


print_stuff.print_table(stats)

Fun.plot(history, print_stuff.plot_options())
