import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jax import jit
from functools import partial
import tensorflow as tf
from utils.print_stuff import plot_options, key_to_tex

import matplotlib.pyplot as plt
import matplotlib.font_manager

matplotlib.rc("font", **{"family": "TeX Gyre DejaVu Math"})
plt.rc("text", usetex=True)


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

        self.define_model()

    def init_model(self):
        key = jax.random.PRNGKey(self.seed_model_init)
        variables = self.model.init(key, jnp.ones(shape=(2,)))
        self.params = variables['params']
        self.batch_stats = variables['batch_stats']
        self.flatened_params, self.deflat_params = ravel_pytree(self.params)

    def init_iterator(self):
        self.train_ds_iterator = (
            self.train_ds.shuffle(buffer_size=self.train_ds.cardinality(), seed=self.seed_iterator)
            .padded_batch(self.batch_size, drop_remainder=True)
            .repeat(self.n_epochs)
            .prefetch(tf.data.AUTOTUNE)
            .as_numpy_iterator()
        )

    def init_params_batch_stats(self):
        self.init_model()
        return self.params, self.batch_stats

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
    def loss_accuracy_batch_stats(self, params, batch_stats, inputs, labels):
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
        grad_fn = jax.value_and_grad(self.loss_accuracy_batch_stats, argnums=0, has_aux=True)
        (loss, (accuracy, batch_stats)), grads = grad_fn(params, batch_stats, inputs, labels)
        return loss, accuracy, batch_stats, grads

    @partial(jit, static_argnums=(0,))
    def model_loss(self, params, batch_stats, inputs, labels):
        results, updates = self.model.apply(
            {'params': params, 'batch_stats': batch_stats},
            x=inputs,
            train=True,
            mutable=['batch_stats'],
        )
        loss = self.loss(results, labels, True)
        return loss

    @partial(jit, static_argnums=(0,))
    def grad_model_loss(self, params, batch_stats, inputs, labels):
        grad = jax.grad(self.model_loss, argnums=0, has_aux=False)
        grads = grad(params, batch_stats, inputs, labels)
        return grads

    @partial(jit, static_argnums=(0,))
    def hvp_model_loss(self, params, batch_stats, inputs, labels, v):
        # Compute the Hessian-vector product, with hessian with respect to params
        def grads(params):
            return self.grad_model_loss(params, batch_stats, inputs, labels)

        df, ddfv = jax.jvp(grads, (params,), (v,))
        return ddfv

    def df(self, params):
        return self.grad_model_loss(params, self.batch_stats, self.batch['inputs'], self.batch['labels'])

    def ddfv(self, params, v):
        return self.hvp_model_loss(params, self.batch_stats, self.batch['inputs'], self.batch['labels'], v)

    def loss_accuracy_batch_stats_grads(self):
        return self.grad_fn(self.params, self.batch_stats, self.batch['inputs'], self.batch['labels'])

    def loss_accuracy_batch_stats_grads_given_params(self, params):
        return self.grad_fn(params, self.batch_stats, self.batch['inputs'], self.batch['labels'])

    def train_metrics(self):
        loss, accuracy = self.metrics(self.params, self.batch_stats, "train")
        return {'train_loss': loss, 'train_accuracy': accuracy}

    def test_metrics(self):
        loss, accuracy = self.metrics(self.params, self.batch_stats, "test")
        return {'test_loss': loss, 'test_accuracy': accuracy}

    def get_train_test_metrics(self):
        return {**self.train_metrics(), **self.test_metrics()}

    @partial(jit, static_argnums=(0, 3))
    def metrics(self, params, batch_stats, ds_str):
        losses = []
        accuracies = []
        if ds_str == "train":
            for batch in (
                self.train_ds.padded_batch(self.batch_size, drop_remainder=False)
                .prefetch(tf.data.AUTOTUNE)
                .as_numpy_iterator()
            ):
                results = self.model.apply(
                    {'params': params, 'batch_stats': batch_stats}, x=batch['inputs'], train=False
                )
                losses.append(self.loss(results, batch['labels'], False))
                accuracies.append(self.accuracy(results, batch['labels'], False))
        elif ds_str == "test":
            for batch in (
                self.test_ds.padded_batch(self.batch_size, drop_remainder=False)
                .prefetch(tf.data.AUTOTUNE)
                .as_numpy_iterator()
            ):
                results = self.model.apply(
                    {'params': params, 'batch_stats': batch_stats}, x=batch['inputs'], train=False
                )
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

    # def get_metrics(self, x):
    #     results = self.model.apply(
    #         {'params': self.params, 'batch_stats': self.batch_stats}, x=self.batch['inputs'], train=False
    #     )
    #     return {
    #         'batch_loss': self.loss(results, self.batch['labels'], True),
    #         'batch_accuracy': self.accuracy(results, self.batch['labels'], True),
    #     }

    def plot_histories(self, histories, keys, axes):
        for key, ax in zip(keys, axes):
            for alg_name, history, color, marker in zip(histories.keys(), histories.values(), *plot_options()):
                if key in history:
                    ax.semilogy(history[key], marker=marker, color=color, fillstyle='none', label=alg_name)
            ax.set_title(key_to_tex(key))
            ax.legend()
