import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jax import jit
from functools import partial
import tensorflow as tf
from utils.print_stuff import plot_options, key_to_tex


class Problem:
    def __init__(self, n_epochs, batch_size, dtype=jnp.float32):
        self.n_epochs = int(n_epochs)
        self.batch_size = batch_size
        self.dtype = dtype

        self.seed_datasets = 1989
        self.seed_iterator = 1999
        self.seed_model_init = 2002

        self.get_datasets(self.seed_datasets)
        self.batch_size = int(min(self.batch_size, self.n_train_samples))
        self.n_batches = self.n_train_samples // self.batch_size

        self.train_batches = (
            self.train_ds.padded_batch(self.batch_size, drop_remainder=False)
            .prefetch(tf.data.AUTOTUNE)
            .as_numpy_iterator()
        )
        self.test_batches = (
            self.test_ds.padded_batch(self.batch_size, drop_remainder=False)
            .prefetch(tf.data.AUTOTUNE)
            .as_numpy_iterator()
        )

        self.define_model()
        self.init_model(self.seed_model_init)

    def init_model(self, seed):
        key = jax.random.PRNGKey(seed)
        variables = self.model.init(key, jnp.ones(shape=(2,)))
        self.params = variables['params']
        self.batch_stats = variables['batch_stats']
        self.flatened_params, self.deflat_params = ravel_pytree(self.params)
        pass

    def init_iterator(self):
        self.train_ds_iterator = (
            self.train_ds.shuffle(buffer_size=self.train_ds.cardinality(), seed=self.seed_iterator)
            .padded_batch(self.batch_size, drop_remainder=True)
            .repeat(self.n_epochs)
            .prefetch(tf.data.AUTOTUNE)
            .as_numpy_iterator()
        )

    def get_params_batch_stats(self):
        return self.params, self.batch_stats

    def get_flattened_params_batch_stats(self):
        return self.flatened_params, self.batch_stats

    def get_model(self):
        return self.model

    def next_batch(self, iter):
        if iter == 0:
            self.init_iterator()
        self.batch = next(self.train_ds_iterator, None)
        if self.batch is None:
            return False
        else:
            return True

    @partial(jit, static_argnums=(0,))
    def train_loss_accuracy_batch_stats(self, params, batch_stats, inputs, labels):
        results, updates = self.model.apply(
            {'params': params, 'batch_stats': batch_stats},
            x=inputs,
            train=True,
            mutable=['batch_stats'],
        )
        loss = self.loss(results, labels, True)
        accuracy = self.accuracy(results, labels, True)
        return loss, (accuracy, updates["batch_stats"])

    @partial(jit, static_argnums=(0,))
    def grad_fn(self, params, batch_stats, inputs, labels):
        grad_fn = jax.value_and_grad(self.train_loss_accuracy_batch_stats, argnums=0, has_aux=True)
        (loss, (accuracy, batch_stats)), grads = grad_fn(params, batch_stats, inputs, labels)
        return loss, accuracy, batch_stats, grads

    def loss_accuracy_batch_stats_grads(self):
        return self.grad_fn(self.params, self.batch_stats, self.batch['inputs'], self.batch['labels'])

    def train_metrics(self):
        loss, accuracy = self.metrics(self.params, self.batch_stats, self.train_batches)
        return {'train_loss': loss, 'train_accuracy': accuracy}

    def test_metrics(self):
        loss, accuracy = self.metrics(self.params, self.batch_stats, self.test_batches)
        return {'test_loss': loss, 'test_accuracy': accuracy}

    @partial(jit, static_argnums=(0, 3))
    def metrics(self, params, batch_stats, batches):
        losses = []
        accuracies = []
        for batch in batches:
            results = self.model.apply({'params': params, 'batch_stats': batch_stats}, x=batch['inputs'], train=False)
            losses.append(self.loss(results, batch['labels'], False))
            accuracies.append(self.accuracy(results, batch['labels'], False))
        loss = jnp.mean(jnp.hstack(losses))
        accuracy = jnp.mean(jnp.hstack(accuracies))

        return loss, accuracy

    @partial(jit, static_argnums=(0, 3))
    def loss(self, results, labels, mean):
        loss = self.loss_el_wise(results, labels)
        if mean:
            loss = jnp.mean(loss)
        return loss

    @partial(jit, static_argnums=(0, 3))
    def accuracy(self, results, labels, mean):
        accuracy = self.accuracy_el_wise(results, labels)
        if mean:
            accuracy = jnp.mean(accuracy)
        return accuracy

    def get_metrics_keys(self):
        return ['batch_loss', 'batch_accuracy', 'train_loss', 'train_accuracy', 'test_loss', 'test_accuracy']

    def get_metrics(self, x):
        results = self.model.apply(
            {'params': self.params, 'batch_stats': self.batch_stats}, x=self.batch['inputs'], train=False
        )
        return {
            'batch_loss': self.loss(results, self.batch['labels'], True),
            'batch_accuracy': self.accuracy(results, self.batch['labels'], True),
        }

    def plot_histories(self, histories, keys, axes):
        for key, ax in zip(keys, axes):
            for alg_name, history, color, marker in zip(histories.keys(), histories.values(), *plot_options()):
                ax.semilogy(history[key], marker=marker, color=color, fillstyle='none', label=alg_name)
            ax.set_title(key_to_tex(key))
            ax.legend()
