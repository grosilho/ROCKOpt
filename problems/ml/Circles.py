import jax
import jax.numpy as jnp
from jax import jit
from jax.flatten_util import ravel_pytree
from functools import partial
import numpy as np
import optax
import tensorflow as tf

from .Problem import Problem
from .NN import ExplicitMLP

tf.config.experimental.set_visible_devices([], 'GPU')


class Circles(Problem):
    def __init__(self, n_train_samples, n_epochs, batch_size):

        self.n_train_samples = int(n_train_samples)
        self.n_test_samples = int(0.25 * n_train_samples)
        super().__init__(n_epochs, batch_size)

    def define_model(self, mp_dtype):
        self.features = [5, 5, 2]
        self.model = dict()
        self.model[mp_dtype.high.dtype] = ExplicitMLP(features=self.features, dtype=mp_dtype.high.dtype)
        if mp_dtype.low is not None:
            self.model[mp_dtype.low.dtype] = ExplicitMLP(features=self.features, dtype=mp_dtype.low.dtype)

    def dataset_generator(self, seed):
        R = [[1.0, 2.0], [3.0, 4.0]]
        key = jax.random.PRNGKey(seed)
        for _ in range(self.n_train_samples + self.n_test_samples):
            key, *subkey = jax.random.split(key, 4)
            label = jax.random.randint(subkey[0], shape=(), minval=0, maxval=2).astype(jnp.int32)
            r = jax.random.uniform(subkey[1], shape=(), minval=R[label][0], maxval=R[label][1], dtype=jnp.float32)
            angle = jax.random.uniform(subkey[2], shape=(), minval=0.0, maxval=2 * np.pi, dtype=jnp.float32)
            point = jnp.array([r * np.cos(angle), r * np.sin(angle)], dtype=jnp.float32)
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
        self.train_ds, self.test_ds = tf.keras.utils.split_dataset(
            ds, left_size=self.n_train_samples, shuffle=True, seed=1989
        )
        assert int(self.train_ds.cardinality()) == self.n_train_samples, "Error in the number of training samples."
        assert int(self.test_ds.cardinality()) == self.n_test_samples, "Error in the number of testing samples."

    def init_model(self, mp_dtype):
        self.define_model(mp_dtype)
        key = jax.random.PRNGKey(self.seed_model_init)
        variables = self.model[mp_dtype.high.dtype].init(key, jnp.ones(shape=(2,), dtype=mp_dtype.high.dtype))
        self.params = variables['params']
        self.batch_stats = variables['batch_stats']
        self.flatened_params, self.deflat_params = ravel_pytree(self.params)

    @partial(jit, static_argnums=(0))
    def loss_el_wise(self, results, labels):
        return optax.softmax_cross_entropy_with_integer_labels(logits=results, labels=labels)

    @partial(jit, static_argnums=(0,))
    def accuracy_el_wise(self, results, labels):
        return jnp.argmax(results, -1) == labels

    def plot(self, histories):
        import matplotlib.pyplot as plt

        plt.set_loglevel(level='warning')
        figsize = (10, 8)
        fig = plt.figure(figsize=figsize)

        axes = fig.subplots(nrows=3, ncols=2)
        self.plot_histories(
            histories,
            keys=["batch_loss", "batch_accuracy", "train_loss", "train_accuracy", "test_loss", "test_accuracy"],
            axes=axes.ravel(),
        )

        plt.show()
