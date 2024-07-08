from flax import linen as nn

from flax.metrics import tensorboard
from flax.training import train_state
import jax
import jax.numpy as jnp
from jax import jit
from functools import partial
import numpy as np
import optax
import tensorflow as tf
from typing import Any, Sequence


class ExplicitMLP(nn.Module):
    features: Sequence[int]
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x, train: bool = True):
        for i, feat in enumerate(self.features):
            x = nn.Dense(feat, dtype=self.dtype, param_dtype=self.dtype)(x)
            x = nn.BatchNorm(
                use_running_average=not train, momentum=0.9, epsilon=1e-5, dtype=self.dtype, param_dtype=self.dtype
            )(x)
            if i != len(self.features) - 1:
                x = nn.relu(x)
        return x
