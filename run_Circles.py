import jax
import jax.numpy as jnp
from flax.metrics import tensorboard
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
from Solvers.ML import SGD, Adam
from Problems.ML import Circles
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


# hyperparameters
learning_rate = 0.1
n_epochs = 10
n_samples = 1280  # then splittend in 75% train and 25% test
batch_size = 128
dtype = jnp.float32

# Create the problem
problem = Circles(n_samples, n_epochs, batch_size, dtype)

# tensordboard stuff
workdir = "tensorboard_logs"
tensorboard_writer = tensorboard.SummaryWriter(workdir)

# Create the optimizer
optimizer = Adam(problem, tensorboard_writer, learning_rate)
optimizer.train()
