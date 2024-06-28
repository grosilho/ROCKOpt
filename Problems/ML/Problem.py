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


class Problem:
    def __init__(self, n_epochs, batch_size, dtype=jnp.float32):
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.dtype = dtype

        seed_datasets = 1989
        seed_iterator = 1999
        seed_model_init = 2002

        self.get_datasets(seed_datasets)
        self.train_ds_iterator = (
            self.train_ds.shuffle(buffer_size=self.train_ds.cardinality(), seed=seed_iterator)
            .padded_batch(self.batch_size, drop_remainder=True)
            .repeat(self.n_epochs)
            .prefetch(tf.data.AUTOTUNE)
            .as_numpy_iterator()
        )

        self.define_model()
        self.init_model(seed_model_init)

    def init_model(self, seed):
        key = jax.random.PRNGKey(seed)
        variables = self.model.init(key, jnp.ones(shape=(2,)))
        self.params = variables['params']
        self.batch_stats = variables['batch_stats']

    def get_params_batch_stats(self):
        return self.params, self.batch_stats

    def get_model(self):
        return self.model

    def next_batch(self):
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

    def loss_accuracy_batch_stats_grads(self, params, batch_stats):
        return self.grad_fn(params, batch_stats, self.batch['inputs'], self.batch['labels'])

    @partial(jit, static_argnums=(0,))
    def test_loss_accuracy(self, params, batch_stats):
        losses = []
        accuracies = []
        for batch in (
            self.test_ds.padded_batch(self.batch_size, drop_remainder=False)
            .prefetch(tf.data.AUTOTUNE)
            .as_numpy_iterator()
        ):
            results = self.model.apply({'params': params, 'batch_stats': batch_stats}, x=batch['inputs'], train=False)
            losses.append(self.loss(results, batch['labels'], False))
            accuracies.append(self.accuracy(results, batch['labels'], False))
        loss = jnp.mean(jnp.hstack(losses))
        accuracy = jnp.mean(jnp.hstack(accuracies))
        metrics = {'loss': loss, 'accuracy': accuracy}
        return metrics

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
