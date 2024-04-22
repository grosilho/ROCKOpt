import logging
import numpy as np


class rho_estimator:
    def __init__(self, n):
        self.logger = logging.getLogger("rho_estimator")

        self.shape = (n,)
        self.fx = np.array(self.shape)
        self.eigvec = np.random.rand(*self.shape)
        self.eigval = 0.0

    def rho(self, f, y, fy=None):
        """
        Estimates spectral radius of df/dy.

        It is a nonlinear power method based on finite differentiation: df/dy(y)*v = f(y+v)-f(y) + O(|v|^2)
        The Rayleigh quotient (dot prod) is replaced (bounded) with an l2-norm.
        The algorithm used is a small change that of
        Sommeijer-Shampine-Verwer, implemented in RKC.
        When a guess is provided, in general it converges in 1-2 iterations.

        Parameters:
        f: function to evaluate
        y: point where to evaluate the function
        fy: f(y) (optional). fy can be None or an already available evaluation of f(y)
        """

        maxiter = 100
        safe = 1.05
        tol = 1e-2
        small = 1e-4
        n_f_eval = 0
        tol_norm = 1e-6

        if fy is None:
            fy = f(y)
            n_f_eval += 1

        eigval = self.eigval
        z = self.eigvec
        y_norm = np.linalg.norm(y) / np.sqrt(len(y))
        z_norm = np.linalg.norm(z) / np.sqrt(len(z))

        # Building the vector z so that the difference z-yn is small
        if y_norm > tol_norm and z_norm > tol_norm:
            # here z -> y+z*|y|*small/|z|
            dzy = y_norm * small
            quot = dzy / z_norm
            z *= quot
            z += y
        elif y_norm > tol_norm:
            # here z-> y*(1+small)
            dzy = y_norm * small
            z = y * (1.0 + small)
        elif z_norm > tol_norm:
            # here z-> z*small/|z|
            dzy = small
            quot = dzy / z_norm
            z *= quot
        else:
            # here z=0 becomes z=random and z = z*small/|z|
            z = np.random.rand(*self.shape)
            dzy = small
            z_norm = np.linalg.norm(z)
            quot = dzy / z_norm
            z *= quot

        """
        Here dzy=|z-y| and z=y+(small perturbation)
        In the following loop dzy=|z-yn| remains true, even with the new z
        """

        eigval_old = 0.0
        eigval_very_old = 0.0
        oscillations_counter = 0
        max_oscillating_iterations = 5

        # Start the power method for non linear operator f
        for iter in range(1, maxiter + 1):
            eigvec = f(z)
            eigvec -= fy
            n_f_eval += 1

            dfzfy = np.linalg.norm(eigvec)

            eigval_very_old = eigval_old
            eigval_old = eigval
            eigval = dfzfy / dzy  # approximation of the Rayleigh quotient (not with dot product but just norms)
            eigval = safe * eigval

            self.logger.debug(f"rho_estimator: iter = {iter}, eigval = {eigval}")

            if abs(eigval - eigval_old) <= eigval * tol:
                # The last perturbation is stored. It will very likely be a
                # good starting point for the next rho call.
                eigvec = z
                eigvec -= y
                break

            if (eigval_old - eigval_very_old) * (eigval_old - eigval) > 0:
                oscillations_counter += 1
                if oscillations_counter > max_oscillating_iterations:
                    self.logger.warning("Spectral radius estimator oscillating, we stop here.")
                    eigvec = z
                    eigvec -= y
                    eigval = max(eigval, eigval_old, eigval_very_old)
                    break
            else:
                oscillations_counter = 0

            if dfzfy != 0.0:
                quot = dzy / dfzfy
                z = y + eigvec * quot
                # z is built such that dzy=|z-yn| is still true
            else:
                raise Exception("Spectral radius estimation error.")

        if iter == maxiter and abs(eigval - eigval_old) > eigval * tol:
            self.logger.warning("Spectral radius estimator did not converge.")

        self.logger.info(f"Estimated rho = {eigval:1.2e} in {iter} iterations.")

        self.eigvec = eigvec
        self.eigval = eigval

        return eigval, n_f_eval
