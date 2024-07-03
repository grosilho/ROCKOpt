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

from .Problem import Problem
from .NN import ExplicitMLP

tf.config.experimental.set_visible_devices([], 'GPU')


class Circles(Problem):
    def __init__(self, n_samples, n_epochs, batch_size, dtype=jnp.float32):

        self.n_samples = int(n_samples)
        super().__init__(n_epochs, batch_size, dtype)

    def define_model(self):
        self.features = [5, 5, 2]
        self.model = ExplicitMLP(features=self.features, dtype=self.dtype)

    def dataset_generator(self, seed):
        R = [[1.0, 2.0], [3.0, 4.0]]
        key = jax.random.PRNGKey(seed)
        for _ in range(self.n_samples):
            key, *subkey = jax.random.split(key, 4)
            label = jax.random.randint(subkey[0], shape=(), minval=0, maxval=2).astype(jnp.int32)
            r = jax.random.uniform(subkey[1], shape=(), minval=R[label][0], maxval=R[label][1], dtype=self.dtype)
            angle = jax.random.uniform(subkey[2], shape=(), minval=0.0, maxval=2 * np.pi, dtype=self.dtype)
            point = jnp.array([r * np.cos(angle), r * np.sin(angle)], dtype=self.dtype)
            yield dict(inputs=point, labels=label)

    def get_datasets(self, seed_datasets):
        ds = tf.data.Dataset.from_generator(
            self.dataset_generator,
            args=[seed_datasets],
            output_signature={
                'inputs': tf.TensorSpec(shape=(2,), dtype=tf.float32),
                'labels': tf.TensorSpec(shape=(), dtype=tf.int32),
            },
        )
        self.train_ds, self.test_ds = tf.keras.utils.split_dataset(ds, left_size=0.75, shuffle=True, seed=1989)
        self.n_train_samples = int(self.train_ds.cardinality())
        self.n_test_samples = int(self.test_ds.cardinality())

    @partial(jit, static_argnums=(0))
    def loss_el_wise(self, results, labels):
        return optax.softmax_cross_entropy_with_integer_labels(logits=results, labels=labels)

    @partial(jit, static_argnums=(0,))
    def accuracy_el_wise(self, results, labels):
        return jnp.argmax(results, -1) == labels

    def plot(self, history):
        import matplotlib.pyplot as plt

        plt.set_loglevel(level='warning')
        figsize = (18, 8)
        fig = plt.figure(figsize=figsize)

        ax1 = fig.add_subplot(321)
        ax2 = fig.add_subplot(322)
        # Loss function + penalization and delta
        self.plot_fx_and_delta(ax1, ax2, history)

        plt.show()
