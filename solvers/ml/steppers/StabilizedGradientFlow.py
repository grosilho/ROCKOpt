import jax
import jax.numpy as jnp
from jax import jit
import tree_math as tm
import tree_math.numpy as tnp

from functools import partial

from solvers.opt.stabilized_utils.rho_estimator import rho_estimator
from solvers.opt.stabilized_utils.es_methods import RKW1, RKC1, RKU1
from utils.HelperClass import HelperClass


class tm_wrapped:
    key = jax.random.key

    def normal_like(key, x):
        xf, unflatten = jax.flatten_util.ravel_pytree(x.tree)
        r = jax.random.normal(key, xf.shape)
        return tm.Vector(unflatten(r))

    def dot(x, y):
        return tnp.dot(x, y)

    def norm(x):
        return jnp.sqrt(tnp.dot(x, x))


class SGF(HelperClass):
    """
    Stabilized Gradient Flow (SGF) method. This method solves the gradient flow ODE using a stabilized method.
    """

    def __init__(self, params):

        default_params = {
            "delta": 0.1,
            "rho_freq": 5,
            "method": "RKC1",
            "damping": 0.05,
            "safe_add": 1,
            "fixed_s": 0,  # 0 ffor automatic selection of s, >0 for fixed s at that value
            "log_history": False,
            "record_stages": False,
            "record_rejected": False,
        }
        super().__init__(default_params, params)

        method_mapping = {
            "RKC1": RKC1,
            "RKW1": RKW1,
            "RKU1": RKU1,
        }

        assert self.method in method_mapping, f"Stabilized method {self.method} not implemented."

        self.es = method_mapping[self.method](self.damping, self.safe_add)

        self.stats_keys = [
            "f_evals",
            "df_evals",
            "ddf_evals",
            "ddf_solves",
            "ddf_mults",
        ]
        self.history_keys = ["delta"]

    def pre_loop(self, params, batch_stats):
        self.init_stats(self.stats_keys)
        self.init_history(self.history_keys)

        self.fx = jnp.inf
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

        loss, accuracy, batch_stats, grads = problemML.loss_accuracy_batch_stats_grads()
        gj = self.small_body(self.es.mu[0], delta, gjm1, grads)
        problemML.params = gj.tree
        problemML.batch_stats = batch_stats

        for j in range(2, self.s + 1):
            gjm2 = gjm1
            gjm1 = gj
            loss, accuracy, batch_stats, grads = problemML.loss_accuracy_batch_stats_grads()
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
