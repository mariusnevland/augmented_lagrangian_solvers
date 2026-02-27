import porepy as pp
import numpy as np
from typing import Optional
from functools import partial
import scipy.sparse as sps
from typing import TypeVar, Union
from matplotlib import pyplot as plt
import logging
logger = logging.getLogger(__name__)
Scalar = pp.ad.Scalar
from porepy.numerics.ad.forward_mode import AdArray
FloatType = TypeVar("FloatType", AdArray, np.ndarray, float)


# The implicit return map method, which is equivalent to an Uzawa algorithm.
# NB! To run a model using this algorithm, you need to call run_implicit_return_map_model().

class ImplicitReturnMap:

    def __init__(self, params):
        super().__init__(params)
        self.total_itr = 0
        self.itr_time_step = []
        self.total_linear_itr = 0
        self.outer_loop_itr = 0
        self.accumulated_outer_loop_itr = 0
        self.itr_time_step_counter = 0

    c_n: float  # Regularization parameter

    c_t: float  # Regularization parameter

    t_n_prev: np.ndarray  # Normal traction from previous step of the outer loop.

    t_t_prev: np.ndarray  # Tangential traction from previous step of the outer loop.

    inner_loop_fail: bool = False  # Indicator for inner loop failure.

    first_itr_stats: int  # Number of iterations to solve the first system in the inner loop.

    def contact_mechanics_normal_constant_irm(self, subdomains: list[pp.Grid]) -> pp.ad.Scalar:

        return pp.ad.Scalar(self.c_n, name="contact_mechanics_normal_constant")

    def contact_mechanics_tangential_constant_irm(self, subdomains: list[pp.Grid]) -> pp.ad.Scalar:

        return pp.ad.Scalar(self.c_t, name="contact_mechanics_tangential_constant")
    
    def normal_fracture_deformation_equation(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.Operator:
        """Regularized version of the normal fracture deformation equation."""

        # Variables
        nd_vec_to_normal = self.normal_component(subdomains)
        # The normal component of the contact traction and the displacement jump.
        t_n: pp.ad.Operator = nd_vec_to_normal @ self.contact_traction(subdomains)
        u_n: pp.ad.Operator = nd_vec_to_normal @ self.displacement_jump(subdomains)

        # Maximum function
        num_cells: int = sum([sd.num_cells for sd in subdomains])
        max_function = pp.ad.Function(pp.ad.maximum, "max_function")
        zeros_frac = pp.ad.DenseArray(np.zeros(num_cells), "zeros_frac")
        t_n_prev_ad = pp.ad.DenseArray(self.t_n_prev, "t_n_prev")

        # The complimentarity condition
        equation: pp.ad.Operator = t_n + max_function(
            pp.ad.Scalar(-1.0) * t_n_prev_ad
            - self.contact_mechanics_normal_constant_irm(subdomains)
            * (u_n - self.fracture_gap(subdomains)),
            zeros_frac,
        )
        equation.set_name("normal_fracture_deformation_equation")
        return equation
    
    def tangential_fracture_deformation_equation(
        self,
        subdomains: list[pp.Grid],
    ) -> pp.ad.Operator:
        """Regularized version of the tangential fracture deformation equation."""

        num_cells = sum([sd.num_cells for sd in subdomains])
        nd_vec_to_tangential = self.tangential_component(subdomains)
        tangential_basis = self.basis(subdomains, dim=self.nd - 1)

        scalar_to_tangential = pp.ad.sum_projection_list(tangential_basis)

        t_t: pp.ad.Operator = nd_vec_to_tangential @ self.contact_traction(subdomains)
        u_t: pp.ad.Operator = nd_vec_to_tangential @ self.plastic_displacement_jump(
            subdomains
        )
        u_t_increment: pp.ad.Operator = pp.ad.time_increment(u_t)

        ones_frac = pp.ad.DenseArray(np.ones(num_cells * (self.nd - 1)))
        zeros_frac = pp.ad.DenseArray(np.zeros(num_cells))

        f_max = pp.ad.Function(self.maximum, "max_function")
        f_norm = pp.ad.Function(partial(pp.ad.l2_norm, self.nd - 1), "norm_function")

        c_num_as_scalar = self.contact_mechanics_tangential_constant_irm(subdomains)
        t_t_prev_ad = pp.ad.DenseArray(self.t_t_prev, "t_t_prev")
        tangential_sum = t_t_prev_ad + (scalar_to_tangential @ c_num_as_scalar) * u_t_increment

        norm_tangential_sum = f_norm(tangential_sum)
        norm_tangential_sum.set_name("norm_tangential")

        b_p = f_max(self.friction_bound(subdomains), zeros_frac)
        b_p.set_name("bp")

        bp_tang = (scalar_to_tangential @ b_p) * tangential_sum

        maxbp_abs = scalar_to_tangential @ f_max(b_p, norm_tangential_sum)

        characteristic = self.contact_mechanics_open_state_characteristic(subdomains)

        equation: pp.ad.Operator = (ones_frac - characteristic) * (
            bp_tang - maxbp_abs * t_t
        ) + characteristic * t_t
        equation.set_name("tangential_fracture_deformation_equation")
        return equation
    
    def maximum(self, var_0: FloatType, var_1: FloatType) -> FloatType:
        # If neither var_0 or var_1 are AdArrays, return the numpy maximum function.
        if not isinstance(var_0, AdArray) and not isinstance(var_1, AdArray):
            # FIXME: According to the type hints, this should not be possible.
            return np.maximum(var_0, var_1)

        # Make a fall-back zero Jacobian for constant arguments.
        # EK: It is not clear if this is relevant, or if we filter out these cases with the
        # above parsing of numpy arrays. Keep it for now, but we should revisit once we
        # know clearer how the Ad-machinery should be used.
        zero_jac = 0
        if isinstance(var_0, AdArray):
            zero_jac = sps.csr_matrix(var_0.jac.shape)
        elif isinstance(var_1, AdArray):
            zero_jac = sps.csr_matrix(var_1.jac.shape)

        # Collect values and Jacobians.
        vals = []
        jacs = []
        for var in [var_0, var_1]:
            if isinstance(var, AdArray):
                v = var.val
                j = var.jac
            else:
                v = var
                j = zero_jac
            vals.append(v)
            jacs.append(j)

        # If both are scalar, return same. If one is scalar, broadcast explicitly
        if isinstance(vals[0], (float, int)):
            if isinstance(vals[1], (float, int)):
                # Both var_0 and var_1 are scalars. Treat vals as a numpy array to return
                # the maximum. The Jacobian of a scalar is 0.
                val = np.max(vals)
                return pp.ad.AdArray(val, 0)
            else:
                # var_0 is a scalar, but var_1 is not. Broadcast to shape of var_1.
                vals[0] = np.ones_like(vals[1]) * vals[0]
        if isinstance(vals[1], (float, int)):
            # var_1 is a scalar, but var_0 is not (or else we would have hit the return
            # statement in the above double-if). Broadcast var_1 to shape of var_0.
            vals[1] = np.ones_like(vals[0]) * vals[1]

        # By now, we know that both vals are numpy arrays. Try to convince mypy that this is
        # the case.
        assert isinstance(vals[0], np.ndarray) and isinstance(vals[1], np.ndarray)
        # Maximum of the two arrays
        inds = (vals[1] >= vals[0]).nonzero()[0]

        max_val = vals[0].copy()
        max_val[inds] = vals[1][inds]
        # If both arrays are constant, a 0 matrix has been assigned to jacs.
        # Return here to avoid calling copy on a number (immutable, no copy method) below.
        if isinstance(jacs[0], (float, int)):
            assert np.isclose(jacs[0], 0)
            assert np.isclose(jacs[1], 0)
            return AdArray(max_val, 0)

        # Start from var_0, then change entries corresponding to inds.
        max_jac = jacs[0].copy()

        if isinstance(max_jac, (sps.spmatrix, sps.sparray)):
            # Enforce csr format, unless the matrix is csc, in which case we keep it.
            if not max_jac.getformat() == "csc":
                max_jac = max_jac.tocsr()
            lines = pp.matrix_operations.slice_sparse_matrix(jacs[1].tocsr(), inds)
            pp.matrix_operations.merge_matrices(max_jac, lines, inds, max_jac.getformat())
        else:
            max_jac[inds] = jacs[1][inds]

        return AdArray(max_val, max_jac)

    def initial_condition(self) -> None:
        super().initial_condition()

        self.c_n_init = self.contact_mechanics_normal_constant(self.mdg.subdomains()).value(self.equation_system)
        self.c_t_init = self.contact_mechanics_tangential_constant(self.mdg.subdomains()).value(self.equation_system)
        self.c_n = self.c_n_init  # Initial numerical constant for the normal equation
        self.c_t = (
            self.c_t_init
        )  # Initial numerical constant for the tangential equation

        # Initial values of normal and tangential traction
        subdomains = self.mdg.subdomains(dim=self.nd - 1)
        t_n = self.normal_component(subdomains) @ self.contact_traction(subdomains)
        t_t = self.tangential_component(subdomains) @ self.contact_traction(subdomains)
        self.t_n_prev = t_n.value(self.equation_system)
        self.t_t_prev = t_t.value(self.equation_system)

    def after_nonlinear_convergence(
        self, iteration_counter: Optional[int] = None
    ) -> None:
        """Some postprocessing is needed before proceeding to the next outer iteration.
        Note also that we do not proceed to the next time step after nonlinear convergence,
        as we also require the outer loop to converge.
        """
        # Update numerical constants before proceeding to the next outer iteration.
        if self.params.get("irm_update_strategy", False):
            if self.c_n >= 1e3:
                self.c_n += 0
                self.c_t += 0
            else:
                self.c_n *= 10
                self.c_t *= 10
        else:
            self.c_n += 0
            self.c_t += 0

        # Update normal and tangential tractions, to be used as constants in the next
        # outer iteration.
        subdomains = self.mdg.subdomains(dim=self.nd - 1)
        t = self.contact_traction(subdomains)
        t_n = self.normal_component(subdomains) @ self.contact_traction(
            subdomains)
        t_t = self.tangential_component(subdomains) @ self.contact_traction(
            subdomains)
        self.t_prev = t.value(self.equation_system)
        self.t_n_prev = t_n.value(self.equation_system)
        self.t_t_prev = t_t.value(self.equation_system)

        # Update equations with the normal and tangential tractions defined above.
        eqn_lst = list(self.equation_system.equations.keys())
        for eqn in eqn_lst:
            self.equation_system.remove_equation(eqn)
        self.set_equations()

        # Keep track of the total number of nonlinear iterations
        # In other words, the total number of linear systems solved.
        self.itr_time_step_counter += self.nonlinear_solver_statistics.num_iteration
        if self.outer_loop_itr == 0:
            self.first_itr_stats = self.nonlinear_solver_statistics.num_iteration
        self.outer_loop_itr += 1
        self.accumulated_outer_loop_itr += 1

    def after_nonlinear_failure(self) -> None:
        """Method to be called if the inner Newton loop fails to converge."""
        if self.time_manager.is_constant:
            # We cannot decrease the constant time step.
            raise ValueError("Inner Newton loop did not converge.")
        else:
            # self.total_itr += self.nonlinear_solver_statistics.num_iteration
            self.itr_time_step_counter += self.nonlinear_solver_statistics.num_iteration
            self.inner_loop_fail = True

    def after_outer_loop_convergence(self, iteration_counter: int) -> None:
        """Method to be called after the return map algorithm has converged.

        Note that this method is nearly identical to the after_nonlinear_convergence()
        method in the case of a standard Newton solver. Hence, we have advanced to the
        next time step when this method is called."""

        solution = self.equation_system.get_variable_values(iterate_index=0)
        # Update the time step magnitude if the dynamic scheme is used.
        # The step size is determined based on the number of iterations
        # needed to solve the first system in the inner loop.
        if not self.time_manager.is_constant:
            self.time_manager.compute_time_step(
                iterations=self.first_itr_stats
            )
        self.update_solution(solution)
        self.convergence_status = True
        self.save_data_time_step()
        # Reset numerical constants, in case an update strategy was used.
        self.c_n = self.c_n_init
        self.c_t = self.c_t_init
        # We do not reset t_n_prev and t_t_prev, as we want the initial
        # guess at the next time step to be the solution at the previous time step.
        self.outer_loop_itr = 0  # Reset outer loop counter for next time step.
        # self.itr_time_step.append(self.itr_time_step_counter)
        self.itr_time_step_counter = 0

    def after_outer_loop_failure(self) -> None:
        """Method to be called if the outer loop fails to converge."""
        self.save_data_time_step()
        if self.time_manager.is_constant or not self.inner_loop_fail:
            raise ValueError("Outer loop iterations did not converge.")
        else:         
            # Update the time step magnitude if the dynamic scheme is used.
            # Note: It will also raise a ValueError if the minimal time step is reached.
            self.time_manager.compute_time_step(recompute_solution=True)
            # Reset the iterate values. This ensures that the initial guess for an
            # unknown time step equals the known time step.
            prev_solution = self.equation_system.get_variable_values(time_step_index=0)
            self.equation_system.set_variable_values(prev_solution, iterate_index=0)
            self.inner_loop_fail = False  # Reset inner loop failure indicator.
            self.itr_time_step_counter = 0


def run_implicit_return_map_model(model, params: dict) -> None:
    """Run a time-dependent model using the implicit return map method. Must be combined
    with a mixin defining the Uzawa equations and solution strategy."""

    # Assign parameters, variables and discretizations. Discretize time-indepedent terms
    if params.get("prepare_simulation", True):
        model.prepare_simulation()

    # Assign a solver for the regularized systems in the inner loop of the return map algorithm.
    solver = _choose_solver(model, params)

    def return_map_algorithm() -> None:
        converged = False
        while not converged:     
            # One iteration of the outer loop consists of replacing the complementarity
            # functions with regularized versions, and solving the resulting nonlinear
            # system.
            # The regularizations depend on the solution at the previous outer iteration.
            val_prev = model.equation_system.get_variable_values(iterate_index=0)
            solver.solve(model)  # Solve regularized nonlinear system.
            if model.inner_loop_fail:
                break  # Exit outer loop if inner loop failed.
            val_current = model.equation_system.get_variable_values(iterate_index=0)

            # Assemble residual of the original, non-regularized contact equations,
            # to be used in the convergence check.
            norm_eqn = \
                pp.contact_mechanics.ContactMechanicsEquations.normal_fracture_deformation_equation\
                    (model, model.mdg.subdomains(dim=model.nd-1))
            tang_eqn = \
                pp.contact_mechanics.ContactMechanicsEquations.tangential_fracture_deformation_equation\
                    (model, model.mdg.subdomains(dim=model.nd-1))
            res_tang = tang_eqn.value(model.equation_system,state=val_current)
            res_norm = norm_eqn.value(model.equation_system,state=val_current)
            residual_contact = np.concatenate((res_norm, res_tang))
            # Residual of the remaining equations.
            residual_non_contact = model.equation_system.assemble(
                equations=["mass_balance_equation",
                           "momentum_balance_equation",
                           "interface_force_balance_equation",
                           "interface_darcy_flux_equation"],
                evaluate_jacobian=False, state=val_current
            )
            residual_total = np.concatenate((residual_contact, residual_non_contact))
            outer_increment = val_current - val_prev
            # Convergence check. We reuse the tolerances for the standard
            # Newton solver.
            outer_increment_norm = model.compute_nonlinear_increment_norm(outer_increment)
            residual_norm = model.compute_residual_norm(residual_total, residual_total)
            # First check if the outer loop diverged to infinity.
            if residual_norm > params["nl_divergence_tol"]:
                break
            tol_outer_increment = params["nl_convergence_tol"]
            tol_residual = params["nl_convergence_tol_res"]
            if outer_increment_norm < tol_outer_increment and residual_norm < tol_residual:
                converged = True
        if converged:
            model.after_outer_loop_convergence(model.outer_loop_itr)
        else:
            model.after_outer_loop_failure()

    # Define a function that does all the work during one time step.
    def time_step() -> None:
        model.time_manager.increase_time()
        model.time_manager.increase_time_index()
        logger.info(
            f"\nTime step {model.time_manager.time_index} at time"
            + f" {model.time_manager.time:.1e}"
            + f" of {model.time_manager.time_final:.1e}"
            + f" with time step {model.time_manager.dt:.1e}"
        )
        return_map_algorithm()

    while not model.time_manager.final_time_reached():
        time_step()
        eqn_lst = list(model.equation_system.equations.keys())
        for eqn in eqn_lst:
            model.equation_system.remove_equation(eqn)
        model.set_equations()

    model.after_simulation()


def _choose_solver(model, params: dict) -> Union[pp.LinearSolver, pp.NewtonSolver]:
    """Choose between linear and non-linear solver.

    Parameters:
        model: Model class containing all information on material parameters, variables,
            discretization and geometry. Various methods such as those relating to solving
            the system, see the appropriate solver for documentation.
        params: Parameters related to the solution procedure.

    """
    if "nonlinear_solver" in params:
        solver = params["nonlinear_solver"](params)
    elif model._is_nonlinear_problem():
        solver = pp.NewtonSolver(params)
    else:
        solver = pp.LinearSolver(params)
    return solver