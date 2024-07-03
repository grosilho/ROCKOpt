import jax
import jax.numpy as jnp
from flax.metrics import tensorboard
import logging

from Solvers.ML import SGD, Adam, TrustRegion, StabilizedGradientFlow
from Problems.ML import Circles
import print_stuff

gpu = True
jit = False
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

# tensordboard stuff
workdir = "tensorboard_logs"
tensorboard_writer = tensorboard.SummaryWriter(workdir)

# hyperparameters
n_epochs = 100
n_samples = 1280  # then splittend in 75% train and 25% test
batch_size = jnp.inf  # if inf, then the whole dataset is used
dtype = jnp.float32

# Create the problem
problem = Circles(n_samples, n_epochs, batch_size, dtype)


# Create the optimizer
# learning_rate = 0.1
# optimizer = Adam(problem, tensorboard_writer, learning_rate)
# optimizer.train()

history = dict()
stats = dict()

common_options = options = {"max_iter": 30, "rtol": 1e-6, "atol": 0.0}

specific_options = dict()
specific_options["TrustRegion"] = {
    "delta_max": 1.0,
    "eta": 1e-4,
    "loc_prob_sol": "dog_leg",
    "method": "direct",
    "iter_solver_tol": 1e-5,
    "iter_solver_maxiter": 100,
}
specific_options["StabilizedGradientFlow"] = {
    "delta_max": 5.0,
    "method": "RKC1",
    "damping": 10.0,
    "safe_add": 1,
    "rho_freq": 10,
    "record_stages": False,
}
# TR = TrustRegion(problem, tensorboard_writer, **common_options, **specific_options["TrustRegion"])
# history[TR.description], stats[TR.description] = TR.train()

SGF = StabilizedGradientFlow(
    problem, tensorboard_writer, **common_options, **specific_options["StabilizedGradientFlow"]
)
history[SGF.description], stats[SGF.description] = SGF.train()

print_stuff.print_table(stats)

problem.plot(history)
