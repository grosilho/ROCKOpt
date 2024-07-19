from jax import jit
from functools import partial
import jax
import jax.numpy as jnp
import scipy

from utils.common import MP_dtype
from utils.HelperClass import HelperClass


class TR(HelperClass):

    default_high_dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
    default_low_dtype = None

    def __init__(self, problem, params):

        default_params = {
            "delta_max": 1.0,
            "delta_init": 0.1,
            "eta": 1e-4,
            "local_problem_solver": "dog_leg",
            "method": "iterative",
            "iter_solver_tol": 1e-5,
            "iter_solver_maxiter": 100,
            "log_history": False,
            "record_rejected": False,
            "dtype": MP_dtype(self.default_high_dtype, self.default_low_dtype),
        }
        super().__init__(default_params, params)

        self.description = "TR"
        self.problem = problem

        assert self.local_problem_solver in ["dog_leg", "cauchy_point"], "Invalid local problem solver"
        self.compute_direction = self.dog_leg if self.local_problem_solver == "dog_leg" else self.cauchy_point

        self.stats_keys = [
            "f_evals",
            "df_evals",
            "ddf_evals",
            "ddf_solves",
            "ddf_mults",
        ]
        self.history_keys = ["delta"]

    def pre_loop(self, x, fx):
        self.init_stats(self.stats_keys)
        self.init_history(self.history_keys)

        self.delta = self.delta_init

        self.pB0 = -self.problem.df(x)
        self.stats["df_evals"] += 1

        if self.log_history:
            self.append_to_history(delta=self.delta)

    def step(self, x, fx):
        """
        Solve minimization problem using trust region method.
        """

        delta = self.delta

        p, gp, pBp = self.compute_direction(x, delta)
        x, fx, rho, accepted, dx_norm, f_diff, model_diff = self.accept_or_reject_step(x, fx, p, gp, pBp)

        if self.log_history and (accepted or self.record_rejected):
            self.append_to_history(delta=delta)

        x_norm = jnp.linalg.norm(x)
        fx_norm = jnp.abs(fx)
        convergence_info = {"f_diff": f_diff, "dx_norm": dx_norm, "x_norm": x_norm, "fx_norm": fx_norm}
        stepper_log_str = f"rho = {rho:.3e}, {u"Δ"} = {delta:.3e}, accepted = {accepted}, dx = {dx_norm:.3e}, df = {f_diff:.3e}, model df = {model_diff:.3e}"

        delta = self.update_delta(f_diff, rho, dx_norm, delta)

        self.delta = delta

        return x, fx, accepted, convergence_info, stepper_log_str

    def accept_or_reject_step(self, x, fx, p, gp, pBp):
        fxp = self.problem.f(x + p)
        self.stats["f_evals"] += 1

        f_diff = fx - fxp
        model_diff = -gp - 0.5 * pBp
        rho = f_diff / model_diff
        dx_norm = jnp.linalg.norm(p)

        accepted = rho > self.eta and f_diff >= 0.0

        if accepted:
            x += p
            fx = fxp

        return x, fx, rho, accepted, dx_norm, f_diff, model_diff

    def update_delta(self, f_diff, rho, dx_norm, delta):
        if f_diff < 0.0:
            delta = 0.5 * delta
        elif rho > 0.75 and jnp.isclose(dx_norm, delta, rtol=1e-3, atol=1e-8):
            delta = min([2 * delta, self.delta_max])
        elif rho < 0.25:
            delta = 0.5 * dx_norm

        return delta

    def dog_leg(self, x, delta):
        """
        Computes the dog-leg step.
        Parameters:
            x: current point
            delta: trust region radius
            p: previous dog-leg step
        Returns:
            p: dog-leg step
            gp: g^T p
            pBp: p^T B p
        """

        pB, g, B = self.newton_direction(self.pB0, x)
        self.pB0 = pB

        if jnp.linalg.norm(pB) <= delta:
            p = pB
        else:
            pU = -jnp.dot(g, g) / jnp.dot(g, B(g)) * g
            if jnp.linalg.norm(pU) >= delta:
                p = delta * pU / jnp.linalg.norm(pU)
            else:
                a = jnp.linalg.norm(pB - pU) ** 2
                b = 2 * jnp.dot(pU, pB - pU)
                c = jnp.linalg.norm(pU) ** 2 - delta**2
                tau = (-b + jnp.sqrt(b**2 - 4 * a * c)) / (2 * a)
                p = pU + tau * (pB - pU)

        return p, jnp.dot(g, p), jnp.dot(p, B(p))

    def newton_direction(self, p, x):
        """
        Compute the Newton direction with a direct or iterative solver.
        """
        g = self.problem.df(x)
        self.stats["df_evals"] += 1

        if self.method == "direct":
            B = self.problem.ddf(x)
            self.stats["ddf_evals"] += 1

            # B_fac = jax.scipy.linalg.cho_factor(B)
            # pB = jax.scipy.linalg.cho_solve(B_fac, -g)
            pB = jax.scipy.linalg.solve(B, -g, lower=True, assume_a='sym')
            self.stats["ddf_solves"] += 1

            def Bv(v):
                self.stats["ddf_mults"] += 1
                return B @ v

        elif self.method == "iterative":
            # solves iteratively in matrix free fashion
            def Bv(v):
                self.stats["ddf_mults"] += 1
                return self.problem.ddfv(x, v)

            # unfortunately, there is no way of counting the number of iterations in jax.scipy.sparse.linalg.bicgstab
            # this is because it does not accept a callback function and moreover the call the the matrix-vector
            # product is compiled, so side-effects as incrementing a counter will not work
            # If you want to count the number of iterations, you can use scipy.sparse.linalg.bicgstab instead, at the price of performance loss.
            # To do so, you can comment the next line and uncomment the other one.
            pB = self.solve_linear_system(x, g, p)
            # pB = self.solve_linear_system_counting(Bv, g, p)

            self.stats["ddf_solves"] += 1
        else:
            raise ValueError("Invalid linear solver")

        return pB, g, Bv

    @partial(jit, static_argnums=(0,))
    def solve_linear_system(self, x, g, p):

        pB, exit_code = jax.scipy.sparse.linalg.bicgstab(
            lambda v: self.problem.ddfv(x, v),
            -g,
            x0=p,
            tol=self.iter_solver_tol,
            atol=1e-8,
            maxiter=self.iter_solver_maxiter,
            M=None,
        )
        return pB

    def solve_linear_system_counting(self, Bv, g, p):
        Bv_op = scipy.sparse.linalg.LinearOperator((p.size, p.size), matvec=Bv)
        pB, exit_code = scipy.sparse.linalg.bicgstab(
            Bv_op,
            -g,
            x0=p,
            rtol=self.iter_solver_tol,
            atol=1e-8,
            maxiter=self.iter_solver_maxiter,
            M=None,
        )
        return pB

    def cauchy_point(self, x, delta):
        """
        Compute the Cauchy point.
        Parameters:
            g: gradient at x
            B: Hessian at x
            delta: trust region radius
        Returns:
            pU: Cauchy point direction
            gp: g^T pU
            pBp: p^UT B pU
        """
        g = self.problem.df(x)
        self.stats["df_evals"] += 1
        pU = -jnp.dot(g, g) / jnp.dot(g, self.problem.ddfv(x, g)) * g
        self.stats["ddf_mults"] += 1
        if jnp.linalg.norm(pU) >= delta:
            pU *= delta / jnp.linalg.norm(pU)
        BpU = self.problem.ddfv(x, pU)
        self.stats["ddf_mults"] += 1
        return pU, jnp.dot(g, pU), jnp.dot(pU, BpU)
