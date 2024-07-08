import jax
import jax.numpy as jnp
from utils.HelperClass import HelperClass


class SF(HelperClass):
    def __init__(self, problem, params):
        default_params = {
            "delta": 0.1,
            "flow": "gradient",
            "linear_solver": "direct",
            "iter_solver_tol": 1e-5,
            "iter_solver_maxiter": 100,
            "log_history": False,
        }
        super().__init__(default_params, params)

        self.problem = problem

        assert self.flow in ["gradient", "newton"], f"Flow {self.flow} not implemented."
        self.description = "GF" if self.flow == "gradient" else "NF"
        self.flow_direction = self.gradient_direction if self.flow == "gradient" else self.newton_direction

        self.stats_keys = [
            "f_evals",
            "df_evals",
            "ddf_evals",
            "ddf_solves",
            "ddf_mults",
        ]
        self.history_keys = []

    def pre_loop(self, x, fx):
        self.init_stats(self.stats_keys)
        self.init_history(self.history_keys)
        self.p = -self.problem.df(x)
        self.stats["df_evals"] += 1

    def step(self, x, fx):

        self.p = self.flow_direction(x, self.p)
        x += self.delta * self.p
        dx_norm = self.delta * jnp.linalg.norm(self.p)

        fx_prev = fx
        fx = self.problem.f(x)
        self.stats["f_evals"] += 1

        f_diff = fx_prev - fx  # this should be positive
        x_norm = jnp.linalg.norm(x)
        fx_norm = jnp.abs(fx)

        accepted = True
        convergence_info = {"f_diff": f_diff, "dx_norm": dx_norm, "x_norm": x_norm, "fx_norm": fx_norm}
        stepper_log_str = f"Δ = {self.delta:.3e}, dx = {dx_norm:.3e}, df = {f_diff:.3e}"

        return x, fx, accepted, convergence_info, stepper_log_str

    def gradient_direction(self, x, p):
        self.stats["df_evals"] += 1
        return -self.problem.df(x)

    def newton_direction(self, x, p):

        g = self.problem.df(x)
        self.stats["df_evals"] += 1

        if self.linear_solver == "direct":
            B = self.problem.ddf(x)
            self.stats["ddf_evals"] += 1

            B_fac = jax.scipy.linalg.cho_factor(B)
            p = jax.scipy.linalg.cho_solve(B_fac, -g)
            self.stats["ddf_solves"] += 1

        elif self.linear_solver == "iterative":

            p, exit_code = jax.scipy.sparse.linalg.bicgstab(
                lambda v: self.problem.ddfv(x, v),
                -g,
                x0=p,
                tol=self.iter_solver_tol,
                atol=1e-8,
                maxiter=self.iter_solver_maxiter,
                M=None,
            )

            self.stats["ddf_solves"] += 1
        else:
            raise ValueError("Invalid linear solver")

        return p
