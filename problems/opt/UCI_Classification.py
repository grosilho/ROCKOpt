import jax
import jax.numpy as jnp
from jax import jit, lax
from jax.flatten_util import ravel_pytree
from flax import linen as nn
from functools import partial
import pandas as pd

from .Problem import Problem


class UCI_Classification(Problem):
    def __init__(self, dataset_name='iris'):

        self.dataset_name = dataset_name

        # Regularization parameter
        self.tau = 0.0

        super().__init__('UCI_Classification')

    def import_data(self):
        from ucimlrepo import fetch_ucirepo

        # fetch dataset
        datasets_ids = {
            "iris": 53,
            "dry_bean": 602,
            "breast_cancer": 17,
            "wine": 109,
            "spam_base": 94,
        }
        dataset = fetch_ucirepo(id=datasets_ids[self.dataset_name])

        # data (as pandas dataframes)
        features = dataset.data.features
        classes = dataset.data.targets

        # convert y to one hot
        features = jnp.array(features.to_numpy(), dtype=jnp.float32)
        classes = jnp.array(pd.get_dummies(classes).to_numpy(), dtype=jnp.int32)
        if classes.shape[1] == 1:
            classes = jnp.hstack((classes, 1 - classes))

        # shuffle rows
        key = jax.random.key(2000)
        perm = jax.random.permutation(key, features.shape[0])
        features = features[perm]
        classes = classes[perm]

        # split data
        n_train = int(0.75 * features.shape[0])
        self.train_features = features[:n_train]
        self.train_classes = classes[:n_train]
        self.test_features = features[n_train:]
        self.test_classes = classes[n_train:]

        # Standardize the data
        mu = jnp.mean(jnp.vstack((self.train_features, self.test_features)), axis=0)
        sigma = jnp.std(jnp.vstack((self.train_features, self.test_features)), axis=0)
        self.train_features = (self.train_features - mu) / sigma
        self.test_features = (self.test_features - mu) / sigma

        # print(f"number of samples: {self.train_features.shape[0]}")
        # print(f"number of features: {self.train_features.shape[1]}")
        # print(f"number of classes: {self.train_classes.shape[1]}")

        self.n_samples = self.train_features.shape[0]

    def define_model(self, mp_dtype):

        # model widths
        self.widths = [self.train_classes[mp_dtype.high.dtype].shape[1]]

        def model(dtype):
            class ExplicitMLP(nn.Module):
                features: list[int]

                def setup(self):
                    self.layers = [nn.Dense(feat, dtype=dtype) for feat in self.features]

                def __call__(self, inputs):
                    x = inputs
                    for i, lyr in enumerate(self.layers):
                        x = lyr(x)
                        if i != len(self.layers) - 1:
                            x = nn.relu(x)
                    return x

            return ExplicitMLP(self.widths)

        self.model = dict()
        self.model[mp_dtype.high.dtype] = model(mp_dtype.high.dtype)
        if mp_dtype.low is not None:
            self.model[mp_dtype.low.dtype] = model(mp_dtype.low.dtype)

    def initial_guess(self, mp_dtype):
        self.import_data()
        self.train_features = {mp_dtype.high.dtype: lax.convert_element_type(self.train_features, mp_dtype.high)}
        self.train_classes = {mp_dtype.high.dtype: lax.convert_element_type(self.train_classes, mp_dtype.high)}
        self.test_features = {mp_dtype.high.dtype: lax.convert_element_type(self.test_features, mp_dtype.high)}
        self.test_classes = {mp_dtype.high.dtype: lax.convert_element_type(self.test_classes, mp_dtype.high)}
        if mp_dtype.low is not None:
            self.train_features[mp_dtype.low.dtype] = lax.convert_element_type(
                self.train_features[mp_dtype.high.dtype], mp_dtype.low
            )
            self.train_classes[mp_dtype.low.dtype] = lax.convert_element_type(
                self.train_classes[mp_dtype.high.dtype], mp_dtype.low
            )
            self.test_features[mp_dtype.low.dtype] = lax.convert_element_type(
                self.test_features[mp_dtype.high.dtype], mp_dtype.low
            )
            self.test_classes[mp_dtype.low.dtype] = lax.convert_element_type(
                self.test_classes[mp_dtype.high.dtype], mp_dtype.low
            )
        self.define_model(mp_dtype)
        key = jax.random.key(1989)
        params = self.model[mp_dtype.high.dtype].init(key, self.train_features[mp_dtype.high.dtype][0])
        flatened_params, self.deflat_params = ravel_pytree(params)
        return flatened_params

    @partial(jit, static_argnums=(0,))
    def f(self, x):
        return self.f_train(x) + self.penalization(x)

    @partial(jit, static_argnums=(0,))
    def f_train(self, x):
        return -jnp.mean(
            self.log_softmax(
                self.model[x.dtype].apply(self.deflat_params(x), self.train_features[x.dtype]),
                self.train_classes[x.dtype],
            )
        )

    @partial(jit, static_argnums=(0,))
    def f_test(self, x):
        return -jnp.mean(
            self.log_softmax(
                self.model[x.dtype].apply(self.deflat_params(x), self.test_features[x.dtype]),
                self.test_classes[x.dtype],
            )
        )

    @partial(jit, static_argnums=(0,))
    def log_softmax(self, scores, classes):
        return jnp.sum(scores * classes, axis=1) - jnp.log(jnp.sum(jnp.exp(scores), axis=1))

    @partial(jit, static_argnums=(0,))
    def penalization(self, x):
        return 0.5 * self.tau * jnp.dot(x, x)

    def accuracy(self, x, features, classes):
        scores = self.model[x.dtype].apply(self.deflat_params(x), features)
        prob = jax.nn.softmax(scores)
        pred = jnp.argmax(prob, axis=1)
        targets = jnp.argmax(classes, axis=1)
        return jnp.mean(pred == targets)

    def get_metrics(self, x):
        return {
            "train_accuracy": self.accuracy(x, self.train_features[x.dtype], self.train_classes[x.dtype]),
            "test_accuracy": self.accuracy(x, self.test_features[x.dtype], self.test_classes[x.dtype]),
            "f_train": self.f_train(x),
            "f_test": self.f_test(x),
        }

    def get_metrics_keys(self):
        return ["train_accuracy", "test_accuracy", "f_train", "f_test"]

    def plot(self, histories):
        import matplotlib.pyplot as plt

        plt.set_loglevel(level='warning')
        figsize = (18, 8)
        fig = plt.figure(figsize=figsize)

        axes = fig.subplots(nrows=3, ncols=2)
        self.plot_histories(
            histories=histories,
            keys=["fx", "delta", "f_train", "f_test", "train_accuracy", "test_accuracy"],
            axes=axes.ravel(),
        )

        plt.show()
