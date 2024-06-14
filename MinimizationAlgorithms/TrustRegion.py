import numpy as np
import scipy
import logging
import time
from .MinimizationAlgorithm import MinimizationAlgorithm


class TrustRegion(MinimizationAlgorithm):
    def __init__(
        self,
        max_iter,
        atol,
        rtol,
        n,
        delta_max,
        eta=1e-4,
        loc_prob_sol="dog_leg",
        method="direct",
        iter_solver_tol=1e-5,
        iter_solver_maxiter=100,
    ):

        description = "TR|" + ("DL" if loc_prob_sol == "dog_leg" else "CP") + "|" + method.capitalize()
        super().__init__(max_iter, atol, rtol, n, description=description)

        self.delta_max = delta_max
        self.eta = eta
        self.local_problem_sol = loc_prob_sol
        self.method = method
        self.iter_solver_tol = iter_solver_tol
        self.iter_solver_maxiter = iter_solver_maxiter

        self.pB0 = np.zeros(n)

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

            Bv_op = scipy.sparse.linalg.LinearOperator((p.size, p.size), matvec=Bv)
            pB, exit_code = scipy.sparse.linalg.cg(
                Bv_op, -g, x0=p, rtol=self.iter_solver_tol, maxiter=self.iter_solver_maxiter, M=None
            )
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

    def solve(self, F, x, delta=0.0, record_rejected=False):
        """
        Solve minimization problem using trust region method.
        Parameters:
        F: function to minimize
        x0: initial guess
        delta: trust region radius. If 0, it will be set to its maximium value
        """

        if delta == 0.0:
            delta = self.delta_max

        p = -F.df(x)  # np.zeros_like(x)

        et = time.process_time()

        fx = F.f(x)
        self.stats["f_evals"] += 1

        self.append_to_history(x, fx, 0.0, True)

        while True:
            if self.local_problem_sol == "dog_leg":
                p, gp, pBp = self.dog_leg(F, x, delta, p)
            elif self.local_problem_sol == "cauchy_point":
                p, gp, pBp = self.cauchy_point(F, x, delta)
            else:
                raise ValueError("Invalid local problem solver")

            fxp = F.f(x + p)
            self.stats["f_evals"] += 1

            true_diff = fx - fxp
            model_diff = -gp - 0.5 * pBp
            rho = true_diff / model_diff

            self.stats["iter"] += 1

            self.logger.info(
                f"Iteration {self.stats["iter"]}: {(f'x = {x.ravel()}, ' if x.size < 5 else "")}f(x) = {F.f(x):.3e}, rho = {rho:.3e}, {u"Δ"} = {delta:.3e}, accepted = {rho > self.eta}"
            )

            if rho > self.eta:
                x += p
                fx = fxp
                self.append_to_history(x, fx, delta, True)
            elif record_rejected:
                self.append_to_history(x + p, fxp, delta, False)

            if rho > 0.75 and np.isclose(np.linalg.norm(p), delta):
                delta = np.min([2 * delta, self.delta_max])
            elif rho < 0.25:
                delta = 0.5 * np.linalg.norm(p)

            if self.check_convergence():
                logging.debug("Convergence reached")
                break

            if self.stats["iter"] >= self.max_iter:
                logging.warning("Maximum number of iterations reached")
                break

        et = time.process_time() - et
        self.stats["cpu_time"] = et
        self.stats["cpu_time_per_iter"] = et / self.stats["iter"]

        self.stats["min_f"] = fx

        return self.history, self.stats
