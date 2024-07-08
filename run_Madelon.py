import logging

from solvers.opt import MinimizationAlgorithm
from solvers.opt.steppers import SGF, TR, STR, SF
from problems.opt import Madelon
from utils.print_stuff import print_table
from utils.common import set_jax_options, initialize_solvers, run_solvers

gpu = True
jit = True
float64 = False
profile = False
log_history = True

set_jax_options(gpu, float64)

logging.basicConfig(level=logging.INFO)

# Define the function to minimize
problem = Madelon()

# The solvers to use
solvers_list = [
    # "SGF",
    # "TR",
    "STR",
    # "NF",
]

# The method used in the outer loop. It will call the steppers
min_algo = MinimizationAlgorithm
min_algo_params = {
    "max_iter": 100,
    "min_iter": 10,
    "rtol": 1e-6,
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
            "delta": 5.0,
            "rho_freq": 5,
            "method": "RKC1",
            "damping": 1.0,
            "safe_add": 1,
            "log_history": log_history,
            "record_stages": False,
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
            "delta_max": 2.0,
            "delta_init": 1.0,
            "eta": 1e-4,
            "local_problem_solver": "dog_leg",
            "method": "iterative",
            "iter_solver_tol": 1e-4,
            "iter_solver_maxiter": 100,
            "log_history": log_history,
            "record_rejected": False,
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
            "delta_max": 2.0,
            "delta_init": 1.0,
            "eta": 1e-4,
            "log_history": log_history,
            "record_rejected": False,
            "method": "RKC1",
            "dt": 2.0,
            "damping": 10.0,
            "safe_add": 2,
            "rho_freq": 1,
            "max_steps": 100,
            "rel_res_tol": 1e-2,
            "abs_tol": 1e-9,
            "rel_tol": 1e-2,
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
            "delta": 0.5,
            "flow": "newton",
            "linear_solver": "iterative",
            "iter_solver_tol": 1e-5,
            "iter_solver_maxiter": 100,
            "log_history": log_history,
        },
    }
)

# Initialize the solvers and perform one iteration to compile the code.
# This is done to avoid the compilation time in the actual simulation.
# Results are discarded.
# Only solvers in solvers_list are initialized and used.
solvers = initialize_solvers(problem, solvers_info, solvers_list)

# Run the solvers (from scratch, i.e. results from the previous iteration are not used).
histories, stats = run_solvers(solvers, jit, profile)

print_table(stats)

if log_history:
    problem.plot(histories)
