from collections import namedtuple
import jax
import jax.numpy as jnp
import logging

# mixed precision dtype
MP_dtype = namedtuple("MP_dtype", ["high", "low"])


def set_jax_options(gpu, float64):
    if not gpu:
        jax.config.update('jax_platform_name', 'cpu')  # Use CPU as default device
    if float64:
        jax.config.update("jax_enable_x64", True)  # Enable 64-bit precision but it is much slower on GPU
    # os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=12'  # Number of cores

    # print(f"Jax Available CPU Devices: {jax.devices("cpu")}")
    # print(f"Jax Available GPU Devices: {jax.devices("gpu")}")
    print(f"Jax Default Backend: {jax.default_backend()}")
    print(f"Jax Default Device: {jnp.ones(3).devices()}")


def initialize_solvers(problem, solvers_info, solvers_list, jit):
    """
    Initialize the solvers and perform one iteration.
    We perform one iteration and discrad the results just to compile the code.
    This is done to avoid the compilation time in the actual simulation
    and thus do not measure compilation time in the performance evaluation.
    """
    solvers = dict()
    log_level = logging.getLogger().getEffectiveLevel()
    for solver_info in solvers_info:
        name = solver_info["name"]
        if name not in solvers_list:
            continue
        min_algo_class = solver_info["min_algo_class"]
        min_algo_params = solver_info["min_algo_params"]
        stepper_class = solver_info["stepper_class"]
        stepper_params = solver_info["stepper_params"]
        solvers[name] = min_algo_class(problem, min_algo_params, stepper_class, stepper_params)
        if jit:
            logging.getLogger().setLevel(logging.ERROR)
            solvers[name].set({"max_iter": 1, "min_iter": 1})
            solvers[name].solve()
            solvers[name].revert(["max_iter", "min_iter"])
            logging.getLogger().setLevel(log_level)

    return solvers


def run_solvers(solvers, jit, profile):
    histories = dict()
    stats = dict()
    # Solve the problem
    with jax.disable_jit(not jit):
        if profile:
            jax.profiler.start_trace("/tmp/jax-trace", create_perfetto_link=True)
        for name, solver in solvers.items():
            histories[name], stats[name] = solver.solve()
        if profile:
            jax.profiler.stop_trace()

    return histories, stats
