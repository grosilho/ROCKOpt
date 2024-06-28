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


class OptaxOptimizer:
    def __init__(self, problem, tensorboard_writer):
        self.problem = problem
        self.tensorboard_writer = tensorboard_writer

    def create_state(self):
        # Define the training state
        class TrainState(train_state.TrainState):
            batch_stats: Any

        model = self.problem.get_model()
        params, batch_stats = self.problem.get_params_batch_stats()
        state = TrainState.create(apply_fn=model.apply, params=params, batch_stats=batch_stats, tx=self.tx)
        return state

    def train_step(self, state):
        """Train for a single step."""
        loss, accuracy, batch_stats, grads = self.problem.loss_accuracy_batch_stats_grads(
            state.params, state.batch_stats
        )
        state = state.apply_gradients(grads=grads)
        state = state.replace(batch_stats=batch_stats)
        metrics = {'loss': loss, 'accuracy': accuracy}
        return state, metrics

    def train(self):
        n_batches = self.problem.n_train_samples // self.problem.batch_size
        i = 1
        state = self.create_state()
        while self.problem.next_batch():

            state, metrics = self.train_step(state)

            epoch = i / n_batches
            print(f"Epoch: {epoch:.2f}, Loss: {metrics['loss']:.3e}, Accuracy: {metrics['accuracy']:.3f}")

            if i % n_batches == 0:
                test_metrics = self.problem.test_loss_accuracy(state.params, state.batch_stats)
                print(
                    f"Epoch: {epoch:.2f}, Test Loss: {test_metrics['loss']:.3e}, Test Accuracy: {test_metrics['accuracy']:.3f}"
                )
                self.tensorboard_writer.scalar("Loss/train", metrics['loss'], i)
                self.tensorboard_writer.scalar("Accuracy/train", metrics['accuracy'], i)
                self.tensorboard_writer.scalar("Loss/test", test_metrics['loss'], i)
                self.tensorboard_writer.scalar("Accuracy/test", test_metrics['accuracy'], i)

            i += 1


class SGD(OptaxOptimizer):
    def __init__(self, problem, tensorboard_writer, learning_rate):
        super().__init__(problem, tensorboard_writer)
        momentum = 0.9
        self.tx = optax.sgd(learning_rate, momentum)


class Adam(OptaxOptimizer):
    def __init__(self, problem, tensorboard_writer, learning_rate):
        super().__init__(problem, tensorboard_writer)
        b1 = 0.9
        b2 = 0.999
        eps = 1e-8
        eps_root = 1e-8
        self.tx = optax.adam(learning_rate, b1=b1, b2=b2, eps=eps, eps_root=eps_root)
