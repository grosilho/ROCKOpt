import jax.numpy as jnp
import logging

import optax
from solvers.ml import Optimizer

from solvers.ml.steppers import OptaxOptimizer, GD, SGF
from problems.ml import Circles
from utils.print_stuff import print_table
from utils.common import set_jax_options, initialize_solvers, run_solvers

gpu = True
jit = True
float64 = False
profile = False
log_history = True

set_jax_options(gpu, float64)

logging.basicConfig(level=logging.INFO)


# hyperparameters
n_epochs = 20
n_samples = 1000  # number of training samples. Another 25% is added for testing
batch_size = 250  # if jnp.inf or >n_samples, then it is set to n_samples and thus the whole dataset is used
dtype = jnp.float32

# Create the problem
problemML = Circles(n_samples, n_epochs, batch_size, dtype)

solvers_list = [
    # "SGD",
    # "Adam",
    "GD",
    "SGF",
]

# The method used in the outer loop. It will call the steppers
optim = Optimizer
optim_params = {
    "max_iter": jnp.inf,
    "min_iter": 1,
    "rtol": 1e-3,
    "atol": 1e-3,
    "log_history": log_history,
    "record_rejected": False,
}

# Define all solvers parameters. Not all those solvers will be used, but only the ones in solvers_list.
solvers_info = []

# The SGD method
solvers_info.append(
    {
        "name": "SGD",
        "min_algo_class": optim,
        "min_algo_params": optim_params,
        "stepper_class": OptaxOptimizer,
        "stepper_params": {
            "optax_optimizer_class": optax.sgd,
            "optax_optimizer_params": {"learning_rate": 2e-1},
        },
    }
)
# The Adam method
solvers_info.append(
    {
        "name": "Adam",
        "min_algo_class": optim,
        "min_algo_params": optim_params,
        "stepper_class": OptaxOptimizer,
        "stepper_params": {
            "optax_optimizer_class": optax.adam,
            "optax_optimizer_params": {
                "learning_rate": 2e-1,
                "b1": 0.9,
                "b2": 0.999,
                "eps": 1e-8,
                "eps_root": 1e-8,
            },
        },
    }
)
# The SGD method, my implementation
solvers_info.append(
    {
        "name": "GD",
        "min_algo_class": optim,
        "min_algo_params": optim_params,
        "stepper_class": GD,
        "stepper_params": {"learning_rate": 10.0},
    }
)
# The SGF method, which is a stabilized method to solve the gradient flow ODE
solvers_info.append(
    {
        "name": "SGF",
        "min_algo_class": optim,
        "min_algo_params": optim_params,
        "stepper_class": SGF,
        "stepper_params": {
            "delta": 10.0,
            "rho_freq": 5 * n_samples // batch_size,
            "method": "RKC1",
            "damping": 0.05,
            "safe_add": 0,
            "fixed_s": 5,
            "log_history": True,
            "record_stages": False,
            "record_rejected": False,
        },
    }
)


# Initialize the solvers and perform one iteration to compile the code.
# This is done to avoid the compilation time in the actual simulation.
# Results are discarded.
# Only solvers in solvers_list are initialized and used.
solvers = initialize_solvers(problemML, solvers_info, solvers_list, jit)

# Run the solvers (from scratch, i.e. results from the previous iteration are not used).
histories, stats = run_solvers(solvers, jit, profile)

print_table(stats)

if log_history:
    problemML.plot(histories)
