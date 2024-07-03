import numpy as np
import scipy
import logging
import time
from .MinimizationAlgorithm import MinimizationAlgorithm


# Helper class to count the number of iterative solver iterations
class solver_matrixvec_mult_counter:
    def __init__(self):
        # Store the number of iterations
        self.niter = 0

    def __call__(self, rk=None):
        # Increment the number of iterations at each call
        self.niter += 1


class TrustRegion(MinimizationAlgorithm):
    def __init__(
        self,
        max_iter,
        atol,
        rtol,
        delta_max,
        eta=1e-4,
        loc_prob_sol="dog_leg",
        method="direct",
        iter_solver_tol=1e-5,
        iter_solver_maxiter=100,
    ):

        description = "TR|" + ("DL" if loc_prob_sol == "dog_leg" else "CP") + "|" + method.capitalize()
        super().__init__(max_iter, atol, rtol, description=description)

        self.delta_max = delta_max
        self.eta = eta
        self.local_problem_sol = loc_prob_sol
        self.method = method
        self.iter_solver_tol = iter_solver_tol
        self.iter_solver_maxiter = iter_solver_maxiter

        self.pB0 = None

        self.logger = logging.getLogger("TrustRegion")

    def dog_leg(self, F, x, delta, p):
        """
        Compute the dog-leg step.
        Parameters:
        F: instance of the function to minimize
        x: current point
        delta: trust region radius
        Returns:
        p: dog-leg step
        gp: g^T p
        pBp: p^T B p
        """
        if self.pB0 is None:
            self.pB0 = -F.df(x)

        pB, g, B = self.newton_direction(self.pB0, F, x)
        self.pB0 = pB

        if np.linalg.norm(pB) <= delta:
            p = pB
        else:
            pU = -np.dot(g, g) / np.dot(g, B(g)) * g
            if np.linalg.norm(pU) >= delta:
                p = delta * pU / np.linalg.norm(pU)
            else:
                a = np.linalg.norm(pB - pU) ** 2
                b = 2 * np.dot(pU, pB - pU)
                c = np.linalg.norm(pU) ** 2 - delta**2
                tau = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)
                p = pU + tau * (pB - pU)

        return p, np.dot(g, p), np.dot(p, B(p))

    def newton_direction(self, p, F, x):
        """
        Compute the Newton direction with a direct or iterative solver.
        """
        g = F.df(x)
        self.stats["df_evals"] += 1

        if self.method == "direct":
            B = F.ddf(x)
            self.stats["ddf_evals"] += 1

            def Bv(v):
                self.stats["ddf_mults"] += 1
                return B @ v

            pB = np.linalg.solve(B, -g)
            self.stats["ddf_solves"] += 1

        elif self.method == "iterative":
            # solve iteratively in matrix free fashion
            def Bv(v):
                self.stats["ddf_mults"] += 1
                return F.ddfv(x, v)

            matvec_counter = solver_matrixvec_mult_counter()
            Bv_op = scipy.sparse.linalg.LinearOperator((p.size, p.size), matvec=Bv)
            pB, exit_code = scipy.sparse.linalg.cg(
                Bv_op,
                -g,
                x0=p,
                rtol=self.iter_solver_tol,
                maxiter=self.iter_solver_maxiter,
                M=None,
                callback=matvec_counter,
            )
            self.stats["ddf_mults"] = matvec_counter.niter
            self.stats["ddf_solves"] += 1
        else:
            raise ValueError("Invalid linear solver")

        return pB, g, Bv

    def cauchy_point(self, F, x, delta):
        """
        Compute the Cauchy point.
        Parameters:
        g: gradient at x
        B: Hessian at x
        delta: trust region radius
        Returns:
        pU: Cauchy point direction
        """
        g = F.df(x)
        self.stats["df_evals"] += 1
        pU = -np.dot(g, g) / np.dot(g, F.ddfv(x, g)) * g
        self.stats["ddf_mults"] += 1
        if np.linalg.norm(pU) >= delta:
            pU *= delta / np.linalg.norm(pU)
        BpU = F.ddfv(x, pU)
        self.stats["ddf_mults"] += 1
        return pU, np.dot(g, pU), np.dot(pU, BpU)

    def solve(self, F, x, delta=None, record_rejected=False, max_iter=None):
        """
        Solve minimization problem using trust region method.
        Parameters:
        F: function to minimize
        x0: initial guess
        delta: trust region radius. If 0, it will be set to its maximium value
        """

        # Clean stats and history
        self.init_stats()
        self.init_history()

        if max_iter is None:
            max_iter = self.max_iter

        if delta is None:
            delta = self.delta_max

        p = -F.df(x)
        self.stats["df_evals"] += 1

        et = time.time()

        fx = F.f(x)
        self.stats["f_evals"] += 1

        self.append_to_history(x, fx, delta, True)

        while True:

            p, gp, pBp = self.compute_direction(F, x, delta, p)

            rho, x, fx, accepted, norm_p, f_diff, model_diff = self.accept_or_reject_step(
                F, x, delta, fx, p, gp, pBp, record_rejected
            )

            self.log(x, fx, rho, delta, accepted, norm_p, f_diff, model_diff)

            delta = self.update_delta(f_diff, rho, norm_p, delta)

            if self.check_convergence(np.abs(f_diff), norm_p, max_iter):
                break

        et = time.time() - et
        self.stats["cpu_time"] = et
        self.stats["cpu_time_per_iter"] = et / self.stats["iter"]

        self.stats["min_f"] = fx

        return self.history, self.stats

    def compute_direction(self, F, x, delta, p):
        if self.local_problem_sol == "dog_leg":
            p, gp, pBp = self.dog_leg(F, x, delta, p)
        elif self.local_problem_sol == "cauchy_point":
            p, gp, pBp = self.cauchy_point(F, x, delta)
        else:
            raise ValueError("Invalid local problem solver")

        return p, gp, pBp

    def accept_or_reject_step(self, F, x, delta, fx, p, gp, pBp, record_rejected):
        fxp = F.f(x + p)
        self.stats["f_evals"] += 1

        f_diff = fx - fxp
        model_diff = -gp - 0.5 * pBp
        rho = f_diff / model_diff

        self.stats["iter"] += 1

        accepted = rho > self.eta and f_diff >= 0.0

        if accepted:
            x += p
            fx = fxp
            self.append_to_history(x, fx, delta, True)
        elif record_rejected:
            self.append_to_history(x + p, fxp, delta, False)

        norm_p = np.linalg.norm(p)

        return rho, x, fx, accepted, norm_p, f_diff, model_diff

    def log(self, x, fx, rho, delta, accepted, norm_p, f_diff, model_diff):
        self.logger.info(
            f"Iteration {self.stats["iter"]}: {(f'x = {x.ravel()}, ' if x.size < 5 else "")}f(x) = {fx:.3e}, rho = {rho:.3e}, {u"Δ"} = {delta:.3e}, accepted = {accepted}"
            + f", dx = {norm_p:.3e}, df = {f_diff:.3e}, model df = {model_diff:.3e}"
        )

    def update_delta(self, f_diff, rho, norm_p, delta):
        if f_diff < 0.0:
            delta = 0.5 * delta
        elif rho > 0.75 and np.isclose(norm_p, delta):
            delta = np.min([2 * delta, self.delta_max])
        elif rho < 0.25:
            delta = 0.5 * norm_p

        return delta
