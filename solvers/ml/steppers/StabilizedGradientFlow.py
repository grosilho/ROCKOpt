import jax
import jax.numpy as jnp
from jax import jit, lax
import tree_math as tm
import tree_math.numpy as tnp

from functools import partial

from utils.common import MP_dtype
from solvers.opt.stabilized_utils.rho_estimator import rho_estimator
from solvers.opt.stabilized_utils.es_methods import RKW1, RKC1, RKU1
from utils.HelperClass import HelperClass


class tm_wrapped:
    key = jax.random.key

    def normal_like(key, x):
        xf, unflatten = jax.flatten_util.ravel_pytree(x.tree)
        r = jax.random.normal(key, xf.shape, dtype=xf.dtype)
        return tm.Vector(unflatten(r))

    def dot(x, y):
        return tnp.dot(x, y)

    def norm(x):
        return jnp.sqrt(tnp.dot(x, x))


class SGF(HelperClass):
    """
    Stabilized Gradient Flow (SGF) method. This method solves the gradient flow ODE using a stabilized method.
    """

    default_high_dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
    default_low_dtype = None

    def __init__(self, params):

        super().__init__(self.default_params, params)

        method_mapping = {
            "RKC1": RKC1,
            "RKW1": RKW1,
            "RKU1": RKU1,
        }

        assert self.method in method_mapping, f"Stabilized method {self.method} not implemented."

        self.es = method_mapping[self.method](self.damping, self.safe_add, dtype=self.dtype.high)

        self.stats_keys = [
            "f_evals",
            "df_evals",
            "ddf_evals",
            "ddf_solves",
            "ddf_mults",
        ]
        self.history_keys = ["delta"]

    @property
    def default_params(self):
        return {
            "delta": 0.1,
            "rho_freq": 5,
            "method": "RKC1",
            "damping": 0.05,
            "safe_add": 1,
            "fixed_s": 0,  # 0 for automatic selection of s, >0 for fixed s at that value
            "log_history": False,
            "record_stages": False,
            "record_rejected": False,
            "dtype": MP_dtype(self.default_high_dtype, self.default_low_dtype),
        }

    def pre_loop(self, params, batch_stats):
        self.init_stats(self.stats_keys)
        self.init_history(self.history_keys)

        self.last_rho_eval_counter = 0
        if self.fixed_s == 0:
            self.rho_old = 0.0
            self.s_old = 0
            self.rho_estimator = rho_estimator(tm.Vector(params), vector_module=tm_wrapped)
        else:
            self.s = self.fixed_s
            self.rho = 0.0  # dummy value
            self.es.update_coefficients(self.s)

    def step(self, problemML):
        """
        Solve the gradient flow ODE
        x'= -F.df(x)
        using the stabilized method.
        """

        self.problemML = problemML

        self.update_rho_and_stages(problemML, self.delta)
        metrics = self.stabilized_iteration(problemML, self.delta)
        accepted = True

        if self.log_history and (accepted or self.record_rejected):
            self.append_to_history(delta=self.delta)

        stepper_log_str = f"eigval = {self.rho:.3e}, Δ = {self.delta:.3e}, s = {self.s}"

        return metrics, accepted, stepper_log_str

    def update_rho_and_stages(self, problemML, delta):
        # re-estimate rho every rho_freq iterations
        if self.fixed_s == 0 and self.last_rho_eval_counter % self.rho_freq == 0:
            # self.rho, n_f_eval = self.rho_estimator.rho_linear_power_method(
            #     lambda v: tm.Vector(problemML.ddfv(problemML.params, v.tree))
            # )

            self.rho, n_f_eval = self.rho_estimator.rho(
                lambda x: tm.Vector(problemML.df(x.tree)), tm.Vector(problemML.params)
            )
            self.stats["df_evals"] += n_f_eval

            # update coefficients if rho has changed
            if self.rho != self.rho_old:
                self.rho_old = self.rho
                self.s = self.es.get_s(delta * self.rho)
                if self.s != self.s_old:
                    self.s_old = self.s
                    self.es.update_coefficients(self.s)

        self.last_rho_eval_counter += 1

    def stabilized_iteration(self, problemML, delta):

        gjm1 = tm.Vector(problemML.params)

        assert gjm1.dtype is self.dtype.high.dtype, "params.dtype != self.dtype.high"

        grads, (loss, accuracy, batch_stats) = problemML.grads_loss_accuracy_batch_stats(self.dtype.high.dtype)
        gj = self.small_body(self.es.mu[0], delta, gjm1, grads)
        problemML.params = gj.tree
        problemML.batch_stats = batch_stats

        for j in range(2, self.s + 1):
            gjm2 = gjm1
            gjm1 = gj
            grads, (loss, accuracy, batch_stats) = problemML.grads_loss_accuracy_batch_stats(self.dtype.high.dtype)
            gj = self.body(self.es.nu[j - 1], self.es.kappa[j - 1], self.es.mu[j - 1], delta, gjm1, gjm2, grads)
            problemML.params = gj.tree
            problemML.batch_stats = batch_stats

        metrics = {'batch_loss': loss, 'batch_accuracy': accuracy}
        self.stats["f_evals"] += self.s
        self.stats["df_evals"] += self.s

        return metrics

    @partial(jit, static_argnums=(0,))
    def body(self, nu, kappa, mu, delta, gjm1, gjm2, grads):
        gj = nu * gjm1 + kappa * gjm2 - mu * delta * tm.Vector(grads)
        return gj

    @partial(jit, static_argnums=(0,))
    def small_body(self, mu, delta, gjm1, grads):
        gj = gjm1 - mu * delta * tm.Vector(grads)
        return gj


class MPSGF(SGF):

    default_high_dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
    default_low_dtype = jnp.float16

    def stabilized_iteration(self, problemML, delta):

        def tree_to_low(tr):
            return jax.tree.map(lambda x: lax.convert_element_type(x, self.dtype.low), tr)

        # try to remove the copy
        params0_vector = tm.Vector(problemML.params.copy())
        params0_vector_low = tm.Vector(tree_to_low(problemML.params))

        djm1 = tnp.zeros_like(tm.Vector(problemML.params))

        assert djm1.dtype is self.dtype.high.dtype, "params.dtype != self.dtype.high"
        assert params0_vector_low.dtype is self.dtype.low.dtype, "params_low.dtype != self.dtype.low"

        grads0, (loss, accuracy, batch_stats) = problemML.grads_loss_accuracy_batch_stats(self.dtype.high.dtype)
        grads0 = tm.Vector(grads0)
        dj = -self.es.mu[0] * delta * grads0

        for j in range(2, self.s + 1):
            djm2 = djm1
            djm1 = dj
            problemML.params = (params0_vector_low + tm.Vector(tree_to_low(dj.tree))).tree
            problemML.batch_stats = batch_stats  # tree_to_low(batch_stats)
            hvp, (grads, loss, accuracy, batch_stats) = problemML.hvp_grads_loss_accuracy_batch_stats(
                tree_to_low(djm1.tree), self.dtype.low.dtype
            )
            dj = self.body(self.es.nu[j - 1], self.es.kappa[j - 1], self.es.mu[j - 1], delta, djm1, djm2, grads0, hvp)

        problemML.params = (params0_vector + dj).tree
        problemML.batch_stats = batch_stats

        metrics = {'batch_loss': loss, 'batch_accuracy': accuracy}
        self.stats["f_evals"] += self.s
        self.stats["df_evals"] += self.s

        return metrics

    @partial(jit, static_argnums=(0,))
    def body(self, nu, kappa, mu, delta, gjm1, gjm2, grads, hvp):
        gj = nu * gjm1 + kappa * gjm2 - mu * delta * (grads + tm.Vector(hvp))
        return gj
