import numpy as np
import porepy as pp
import logging
from typing import Union
logger = logging.getLogger(__name__)


def run_time_dependent_uzawa_model(model, params: dict) -> None:
    """Run a time-dependent model using an Uzawa algorithm. Must be combined
    with a mixin defining the Uzawa equations and solution strategy."""

    # Assign parameters, variables and discretizations. Discretize time-indepedent terms
    if params.get("prepare_simulation", True):
        model.prepare_simulation()

    # Assign a solver for the regularized systems in the inner loop of the Uzawa algorithm.
    solver = _choose_solver(model, params)

    def uzawa_algorithm() -> None:
        converged = False
        max_uzawa_itr = 150  # Maximum number of outer Uzawa iterations
        total_itr = 0
        while not converged and model.uzawa_itr <= max_uzawa_itr:     
            # One iteration of the Uzawa loop consists of replacing the complementarity
            # functions with regularized versions, and solving the resulting nonlinear
            # system.
            # The regularizations depend on the solution at the previous Uzawa iteration.
            val_prev = model.equation_system.get_variable_values(iterate_index=0)
            solver.solve(model)  # Solve regularized nonlinear system.
            total_itr += model.nonlinear_solver_statistics.num_iteration
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
            uzawa_increment = val_current - val_prev
            # Convergence check. We reuse the tolerances for the standard
            # Newton solver.
            uzawa_increment_norm = model.compute_nonlinear_increment_norm(uzawa_increment)
            residual_contact_norm = model.compute_residual_norm(residual_contact, residual_contact)
            # print(residual_contact_norm)
            # First check if the Uzawa loop diverged to infinity.
            div_tol = 1e8
            if residual_contact_norm > div_tol:
                break
            tol_uzawa_increment = params["nl_convergence_tol"]
            tol_residual_contact = params["nl_convergence_tol_res"]
            if uzawa_increment_norm < tol_uzawa_increment and residual_contact_norm < tol_residual_contact:
                converged = True

        if converged:
            model.after_uzawa_convergence(model.uzawa_itr)
        else:
            model.after_uzawa_failure()
        # print("Uzawa iteration counter:", model.uzawa_itr)
        # print("Total iteration counter:", model.total_itr)

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
        uzawa_algorithm()

    while not model.time_manager.final_time_reached():
        time_step()

        # TODO: Is it necessary to reassemble the equations here?
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