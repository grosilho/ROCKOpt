import jax
import jax.numpy as jnp
from flax.metrics import tensorboard
import logging

from Solvers.ML import SGD, Adam, TrustRegion
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

TR_options = {
    "max_iter": 1e2,
    "rtol": 1e-4,
    "atol": 0.0,
    "delta_max": 2.0,
    "eta": 1e-4,
    "loc_prob_sol": "dog_leg",
    "method": "iterative",
    "iter_solver_tol": 1e-4,
    "iter_solver_maxiter": 100,
}
TR = TrustRegion(problem, tensorboard_writer, TR_options)
history[TR.description], stats[TR.description] = TR.train()

print_stuff.print_table(stats)
