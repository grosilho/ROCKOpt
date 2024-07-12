import logging
import jax
import jax.numpy as jnp
from solvers.opt import MinimizationAlgorithm
from solvers.opt.steppers import SGF, MPSGF, TR, STR, SF
from problems.opt import Denoising
from utils.print_stuff import print_table
from utils.common import set_jax_options, initialize_solvers, run_solvers, MP_dtype

gpu = True
jit = True
float64 = False
profile = False
log_history = True
highest_dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32

set_jax_options(gpu, float64)

logging.basicConfig(level=logging.INFO)

# Define the function to minimize
problem = Denoising()

# The solvers to use
solvers_list = [
    "SGF",
    "MPSGF",
    "TR",
    "STR",
    "NF",
    # "GF",
]

# The method used in the outer loop. It will call the steppers
min_algo = MinimizationAlgorithm
min_algo_params = {
    "max_iter": 1000,
    "min_iter": 5,
    "rtol": 1e-2,
    "atol": 0.0,
    "log_history": log_history,
    "record_rejected": False,
}

# Define all solvers parameters. Not all those solvers will be used, but only the ones in solvers_list.
solvers_info = []

# The SGF method, which is a stabilized method to solve the gradient flow ODE
solvers_info.append(
    {
        "name": "SGF",
        "min_algo_class": min_algo,
        "min_algo_params": min_algo_params,
        "stepper_class": SGF,
        "stepper_params": {
            "delta": 1e5,
            "rho_freq": 1e4,
            "method": "RKC1",
            "damping": 1.0,
            "safe_add": 1,
            "log_history": log_history,
            "record_stages": False,
            "dtype": MP_dtype(highest_dtype, None),
        },
    }
)
# The mixed precision SGF method
solvers_info.append(
    {
        "name": "MPSGF",
        "min_algo_class": min_algo,
        "min_algo_params": min_algo_params,
        "stepper_class": MPSGF,
        "stepper_params": {
            "delta": 5e4,
            "rho_freq": 1e4,
            "method": "RKC1",
            "damping": 1.0,
            "safe_add": 1,
            "log_history": log_history,
            "record_stages": False,
            "dtype": MP_dtype(highest_dtype, jnp.float32),
        },
    }
)
# The trust region method
solvers_info.append(
    {
        "name": "TR",
        "min_algo_class": min_algo,
        "min_algo_params": min_algo_params,
        "stepper_class": TR,
        "stepper_params": {
            "delta_max": 100.0,
            "delta_init": 100.0,
            "eta": 1e-4,
            "local_problem_solver": "dog_leg",
            "method": "iterative",
            "iter_solver_tol": 1e-3,
            "iter_solver_maxiter": 100,
            "log_history": log_history,
            "record_rejected": False,
            "dtype": MP_dtype(highest_dtype, None),
        },
    }
)
# The stabilized trust region method
solvers_info.append(
    {
        "name": "STR",
        "min_algo_class": min_algo,
        "min_algo_params": min_algo_params,
        "stepper_class": STR,
        "stepper_params": {
            "delta_max": 100.0,
            "delta_init": 100.0,
            "eta": 1e-4,
            "log_history": log_history,
            "record_rejected": False,
            "dtype": MP_dtype(highest_dtype, None),
            "method": "RKC1",
            "dt": 5e4,
            "damping": 1.0,
            "safe_add": 1,
            "max_steps": 100,
            "rho_freq": 1e4,
            "rel_res_tol": 1e-2,
            "abs_tol": 1e-9,
            "rel_tol": 1e-6,
        },
    }
)
# The Newton Flow
solvers_info.append(
    {
        "name": "NF",
        "min_algo_class": min_algo,
        "min_algo_params": min_algo_params,
        "stepper_class": SF,
        "stepper_params": {
            "delta": 0.1,
            "flow": "newton",
            "linear_solver": "iterative",
            "iter_solver_tol": 1e-4,
            "iter_solver_maxiter": 100,
            "log_history": log_history,
            "dtype": MP_dtype(highest_dtype, None),
        },
    }
)
# The Gradient Flow
solvers_info.append(
    {
        "name": "GF",
        "min_algo_class": min_algo,
        "min_algo_params": min_algo_params,
        "stepper_class": SF,
        "stepper_params": {
            "delta": 2000,
            "flow": "gradient",
            "log_history": log_history,
            "dtype": MP_dtype(highest_dtype, None),
        },
    }
)

# Initialize the solvers and perform one iteration to compile the code.
# This is done to avoid the compilation time in the actual simulation.
# Results are discarded.
# Only solvers in solvers_list are initialized and used.
solvers = initialize_solvers(problem, solvers_info, solvers_list, jit)

# Run the solvers (from scratch, i.e. results from the previous iteration are not used).
histories, stats = run_solvers(solvers, jit, profile)

print_table(stats)

if log_history:
    problem.plot(histories)
