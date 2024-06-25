import numpy as np
import logging

from .es_methods import RKW1, RKC1, RKU1
from .rho_estimator import rho_estimator
from ..TrustRegion import TrustRegion


class StabilizedTrustRegion(TrustRegion):
    def __init__(
        self,
        max_iter,
        atol,
        rtol,
        delta_max,
        eta=1e-4,
        method="RKC1",
        damping=10.0,
        safe_add=0,
        dt=1.0,
        max_steps=10,
        p_conv_tol=1e-5,
        rho_freq=1,
    ):

        super().__init__(max_iter, atol, rtol, delta_max, eta, loc_prob_sol="dog_leg")
        self.description = "StabTR|" + method

        self.es_params = {
            "dt": dt,
            "max_steps": max_steps,
            "p_conv_tol": p_conv_tol,
            "rho_freq": rho_freq,
        }

        self.es = eval(method)(damping, safe_add)
        self.rho_estimator = None
        self.eigval = 0.0
        self.s = 0
        self.rho_count = 0

        self.logger = logging.getLogger("StabilizedTrustRegion")

    def newton_direction(self, p, F, x):
        """
        Do some steps of size dt on the equation
        p' = -g - B @ p , p(0) = p
        using the explicit stabilized method.
        The equilibrium of this equation is at p'==0, hence p = -B^-1 @ g, i.e. the newton step.
        Thus, doing a few steps on this equation we approximate the newton step.
        Compared to standard method:
        Advantage: no need to invert B.
        Disadvantages: need to estimate the spectral radius of B and compute some matrix-vector products.
        """

        g = F.df(x)
        self.stats["df_evals"] += 1

        def f(y):
            self.stats["ddf_mults"] += 1
            return -g - F.ddfv(x, y)

        res0 = np.linalg.norm(f(p))

        dt = self.es_params["dt"]
        max_steps = self.es_params["max_steps"]
        p_conv_tol = self.es_params["p_conv_tol"]
        eigval = self.eigval
        s = self.s

        if self.rho_estimator is None:
            self.rho_estimator = rho_estimator(x.size)

        i = 0
        while True:

            fp = f(p)

            # Note that this if can be executed only at first iteration (i=0)
            if self.rho_count % self.es_params["rho_freq"] == 0 and i == 0:
                # eigval, n_f_eval = self.rho_estimator.rho(f, p, fp)
                eigval, n_f_eval = self.rho_estimator.rho_linear_power_method(lambda v: F.ddfv(x, v))
            if eigval != self.eigval:
                self.eigval = eigval
                s = self.es.get_s(dt * eigval)
                if s != self.s:
                    self.s = s
                    self.es.update_coefficients(s)

            djm2, djm1 = 0.0, 0.0
            dj = self.es.mu[0] * dt * fp
            for j in range(2, s + 1):
                djm2, djm1, dj = djm1, dj, djm2
                dj = self.es.nu[j - 1] * djm1 + self.es.kappa[j - 1] * djm2 + self.es.mu[j - 1] * dt * f(p + djm1)

            p = p + dj

            res = f(p)

            i += 1

            if np.linalg.norm(res) / res0 < p_conv_tol:
                break

            if i >= max_steps:
                self.logger.warning("Stabilized Newton direction reached maximum iterations.")
                break

        self.rho_count += 1

        self.logger.info(f"Stabilized Newton direction with s={s} and rho={eigval:.2f}, converged in {i} iterations.")

        def Bv(v):
            self.stats["ddf_mults"] += 1
            return F.ddfv(x, v)

        return p, g, Bv
