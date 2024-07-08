import logging
import jax
import jax.numpy as jnp
import time
import scipy.sparse.linalg


class rho_estimator:
    def __init__(self, n):
        self.logger = logging.getLogger("rho_estimator")

        key = jax.random.key(1989)
        self.eigvec = jax.random.normal(key, shape=(n,))
        self.eigval = 0.0

    def rho_linear_scipy(self, A):
        """
        Estimate the spectral radius of a linear operator A with scipy.sparse.linalg.eigs.

        Args:
            A: A is a function that takes a vector and returns a vector.

        Returns:
            The spectral radius of A.
        """

        tol = 1e-4
        eigvals, eigvecs = scipy.sparse.linalg.eigs(
            A, k=1, v0=self.eigvec, which="LM", return_eigenvectors=True, tol=tol
        )
        self.eigval = eigvals[0].real
        self.eigvec = eigvecs[:, 0].real

        return self.eigval

    def rho_linear_power_method(self, A):
        """
        Estimate the spectral radius of a linear operator A with power method.
        Args:
            A: A is a function that takes a vector and returns a vector.

        Returns:
            The spectral radius of A.
        """

        et = time.time()

        maxiter = 100
        tol = 1e-4
        n_f_eval = 0

        eigval_old = self.eigval
        eigvec_old = self.eigvec

        # Start the power method for linear operator A
        for iter in range(1, maxiter + 1):

            eigvec = A(eigvec_old)
            eigvec /= jnp.linalg.norm(eigvec)
            eigval = jnp.dot(eigvec, A(eigvec))  # Rayleigh quotient

            n_f_eval += 1

            self.logger.debug(f"rho_estimator: iter = {iter}, eigval = {eigval}")

            if abs(eigval - eigval_old) <= eigval * tol:
                break

            eigval_old = eigval
            eigvec_old = eigvec

        if iter == maxiter and abs(eigval - eigval_old) > eigval * tol:
            self.logger.warning("Spectral radius estimator did not converge.")

        et = time.time() - et

        self.logger.info(f"Estimated rho = {eigval:1.2e} in {iter} iterations and {et:1.2e} seconds.")

        self.eigvec = eigvec
        self.eigval = eigval

        return eigval, n_f_eval

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
        tol = 1e-4
        small = 1e-6
        n_f_eval = 0
        tol_norm = 1e-6

        if fy is None:
            fy = f(y)
            n_f_eval += 1

        eigval = self.eigval
        z = self.eigvec
        y_norm = jnp.linalg.norm(y) / jnp.sqrt(len(y))
        z_norm = jnp.linalg.norm(z) / jnp.sqrt(len(z))

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
            z = jnp.random.rand(*self.shape)
            dzy = small
            z_norm = jnp.linalg.norm(z)
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

            dfzfy = jnp.linalg.norm(eigvec)

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
