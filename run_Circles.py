import jax.numpy as jnp
import logging
from flax.metrics import tensorboard

from problems.ML import Circles
from utils.print_stuff import print_table
from utils.common import set_jax_options, initialize_solvers, run_solvers

gpu = True
jit = True
float64 = False
profile = False

set_jax_options(gpu, float64)

logging.basicConfig(level=logging.INFO)

# tensordboard stuff
workdir = "tensorboard_logs"
tensorboard_writer = tensorboard.SummaryWriter(workdir)

# hyperparameters
n_epochs = 30
n_samples = 128  # then splittend in 75% train and 25% test
batch_size = jnp.inf  # if inf, then the whole dataset is used
dtype = jnp.float32

# Create the problem
problem = Circles(n_samples, n_epochs, batch_size, dtype)

solvers_names = [
    "SGD",
    # "Adam",
    # "TrustRegionML",
    # "StabilizedGradientFlowML",
    "ExactGradientFlowML",
]

history = dict()
stats = dict()

common_options = {
    "max_iter": 100,
    "min_iter": 100,
    "rtol": 1e-3,
    "atol": 0.0,
    "log_history": True,
    "tensorboard_writer": tensorboard_writer,
}

specific_options = dict()
specific_options["SGD"] = {"learning_rate": 0.2}
specific_options["Adam"] = {"learning_rate": 0.2}
specific_options["TrustRegionML"] = {
    "delta_max": 1.0,
    "eta": 1e-4,
    "loc_prob_sol": "dog_leg",
    "method": "iterative",
    "iter_solver_tol": 1e-5,
    "iter_solver_maxiter": 100,
}
specific_options["StabilizedGradientFlowML"] = {
    "delta_max": 0.1,
    "method": "RKC1",
    "damping": 0.0,
    "safe_add": 0,
    "rho_freq": 1e3,
    "record_stages": False,
}
specific_options["ExactGradientFlowML"] = {
    "delta_max": 0.2,
}


solvers = initialize_solvers(solvers_names, problem, common_options, specific_options)
history, stats = run_solvers(solvers, jit, profile)

print_table(stats)

if common_options["log_history"]:
    problem.plot(history)
