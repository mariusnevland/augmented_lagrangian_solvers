from functools import partial
import numpy as np
import logging
import porepy as pp
Scalar = pp.ad.Scalar
logger = logging.getLogger(__name__)

class CycleCheck:

    def fetch_fracture_vars(self):
        # Manage the states
        traction_states = []
        displacement_jump_states = []
        subdomains = self.mdg.subdomains(dim=self.nd - 1)

        # Variables
        traction_states.append(
            self.contact_traction(subdomains).value(self.equation_system)
        )
        displacement_jump_states.append(
            self.displacement_jump(subdomains).value(self.equation_system)
        )

        return traction_states, displacement_jump_states

    def fetch_fracture_residuals(self):
        # Manage the residuals
        normal_residuals = []
        tangential_residuals = []
        subdomains = self.mdg.subdomains(dim=self.nd - 1)

        normal_residuals.append(
            pp.momentum_balance.MomentumBalance.normal_fracture_deformation_equation(
                self, subdomains
            ).value(self.equation_system)
        )
        tangential_residuals.append(
            pp.momentum_balance.MomentumBalance.tangential_fracture_deformation_equation(
                self, subdomains
            ).value(self.equation_system)
        )

        return normal_residuals, tangential_residuals

    def compute_fracture_states(self, split_output: bool = False) -> list:
        # Compute states of each fracture cell using the displacement increment.
        # Returns a list where:
        # Open=0, Sticking=1, Gliding=2
        subdomains = self.mdg.subdomains(dim=self.nd - 1)
        f_norm = pp.ad.Function(partial(pp.ad.l2_norm, self.nd - 1), "norm_function")
        tangential_basis: list[pp.ad.SparseArray] = self.basis(
            subdomains, dim=self.nd - 1  # type: ignore[call-arg]
        )
        t_n = self.normal_component(subdomains) @ self.contact_traction(subdomains)
        t_t = self.tangential_component(subdomains) @ self.contact_traction(subdomains)
        u_n = self.normal_component(subdomains) @ self.displacement_jump(subdomains)
        u_t = self.tangential_component(subdomains) @ self.displacement_jump(subdomains)
        u_t_increment: pp.ad.Operator = pp.ad.time_increment(u_t)
        c_num_as_scalar = self.contact_mechanics_numerical_constant(subdomains)
        scalar_to_tangential = pp.ad.sum_projection_list(tangential_basis)
        tangential_sum = t_t + (scalar_to_tangential @ c_num_as_scalar) * u_t_increment
        norm_tangential_sum = f_norm(tangential_sum)
        b = (
            Scalar(-1.0)
            * self.friction_coefficient(subdomains)
            * (
                t_n
                + self.contact_mechanics_numerical_constant(subdomains)
                * (u_n - self.fracture_gap(subdomains))
            )
        )
        btang = Scalar(-1.0) * self.friction_coefficient(subdomains) * t_n

        norm_tang_sum_eval = norm_tangential_sum.value(self.equation_system)
        b_eval = b.value(self.equation_system)
        states = []
        tol = self.numerical.open_state_tolerance
        for vals_norm, vals_b in zip(norm_tang_sum_eval, b_eval):
            if vals_b - vals_norm > tol and vals_b > tol:
                states.append(1)  # Stick
            elif vals_b - vals_norm <= tol and vals_b > tol:
                states.append(2)  # Glide
            elif vals_b <= tol:
                states.append(0)  # Open
            else:
                print("Should not get here.")

        # Split combined states vector into subdomain-corresponding vectors
        if split_output:
            split_states = []
            num_cells = []
            for sd in subdomains:
                prev_num_cells = int(sum(num_cells))
                split_states.append(
                    np.array(states[prev_num_cells : prev_num_cells + sd.num_cells])
                )
                num_cells.append(sd.num_cells)
            return split_states
        else:
            return states

    def reset_cycling_analysis(self):
        """Clean up all cached data for cycling analysis."""

        if hasattr(self, "cached_contact_states"):
            del self.cached_contact_states
        if hasattr(self, "cached_contact_vars"):
            del self.cached_contact_vars
        if hasattr(self, "stagnating_states"):
            del self.stagnating_states
        if hasattr(self, "cycling_window"):
            del self.cycling_window
        if hasattr(self, "cached_contact_normal_residuals"):
            del self.cached_contact_normal_residuals
        if hasattr(self, "cached_contact_tangential_residuals"):
            del self.cached_contact_tangential_residuals
        # if hasattr(self, "previous_states"):
        #    assert hasattr(self, "states")
        #    self.previous_states = self.states.copy()

    def check_cycling(self):
        """Check for cycling in contact states."""

        # Initialize cache
        if not hasattr(self, "cached_contact_states"):
            self.cached_contact_states = []
        if not hasattr(self, "cached_contact_vars"):
            self.cached_contact_vars = []
        if not hasattr(self, "cached_contact_normal_residuals"):
            self.cached_contact_normal_residuals = []
        if not hasattr(self, "cached_contact_tangential_residuals"):
            self.cached_contact_tangential_residuals = []
        cycling = False
        cycling_window = 0
        stagnating_states = False

        # Fetch states and variables
        self.states = self.compute_fracture_states(split_output=True)
        vars = self.fetch_fracture_vars()
        normal_residuals, tangential_residuals = self.fetch_fracture_residuals()

        # Determine change in time, i.e., the total difference between states and
        # previous_states
        try:
            total_changes_in_time = np.count_nonzero(
                np.logical_not(
                    np.isclose(
                        np.concatenate(self.states),
                        np.concatenate(self.previous_states),
                    )
                )
            )
        except:
            total_changes_in_time = 0
        logger.info(f"Changes in time: {total_changes_in_time}")

        self.nonlinear_solver_statistics.total_contact_state_changes_in_time = (
            total_changes_in_time
        )

        # Check for stagnation in contact states
        required_length = 12
        if len(self.cached_contact_states) >= required_length:
            stagnating_states = True
            for i in range(required_length):
                if not np.allclose(
                    np.concatenate(self.states),
                    np.concatenate(self.cached_contact_states[-i - 1]),
                ):
                    stagnating_states = False
                    break
        if stagnating_states:
            logger.info(f"Stagnating states detected.")

        elif len(self.cached_contact_states) > 0:
            # Determine detailed contact state changes
            # Determine number of cells in each contact state
            # and the number of changes from state i to j.
            changes = np.zeros((3, 3), dtype=int)
            num_contact_states = np.zeros(3, dtype=int)
            try:
                for i in range(3):
                    num_contact_states[i] = int(
                        np.sum(np.concatenate(self.states) == i)
                    )
                    for j in range(3):
                        changes[i, j] = int(
                            np.sum(
                                np.logical_and(
                                    np.concatenate(self.states) == i,
                                    np.concatenate(self.cached_contact_states[-1]) == j,
                                )
                            )
                        )
            except:
                ...
            logger.info(f"Changes in states: \n{changes}")

            # Count general changes
            try:
                total_changes = np.count_nonzero(
                    np.logical_not(
                        np.isclose(
                            np.concatenate(self.states),
                            np.concatenate(self.cached_contact_states[-1]),
                        )
                    )
                )
            except:
                total_changes = 0
            logger.info(f"Total changes: {total_changes}")

            # Check if changes are small
            if total_changes < 7:
                self.small_changes = True
            else:
                self.small_changes = False

            # Monitor contact state changes
            self.nonlinear_solver_statistics.num_contact_states = (
                num_contact_states.tolist()
            )
            self.nonlinear_solver_statistics.contact_state_changes = changes.tolist()
            self.nonlinear_solver_statistics.total_contact_state_changes = total_changes
            if total_changes > 0:
                self.nonlinear_solver_statistics.last_update_contact_states = (
                    self.nonlinear_solver_statistics.num_iteration
                )
            if hasattr(self, "update_num_contact_states_changes"):
                self.update_num_contact_states_changes()

        # Check for cycling based on closedness of states and variables
        rtol = 1e-2
        for i in range(len(self.cached_contact_states) - 1, 1, -1):
            if self.states != [] and (
                np.allclose(
                    np.concatenate(self.states),
                    np.concatenate(self.cached_contact_states[i]),
                )
                and np.allclose(
                    np.concatenate(vars),
                    np.concatenate(self.cached_contact_vars[i]),
                    rtol=rtol,
                )
                and np.allclose(
                    np.concatenate(self.cached_contact_states[-1]),
                    np.concatenate(self.cached_contact_states[i - 1]),
                )
                and np.allclose(
                    np.concatenate(self.cached_contact_vars[-1]),
                    np.concatenate(self.cached_contact_vars[i - 1]),
                    rtol=rtol,
                )
            ):
                cycling = True
                cycling_window = len(self.cached_contact_states) - i

                logger.info(f"Cycling detected with window {cycling_window}.")

            if cycling:
                break
        if cycling_window > 1:
            self.return_map_on = True
        self.cached_contact_states.append(self.states)
        self.cached_contact_vars.append(vars)
        self.cached_contact_normal_residuals.append(normal_residuals)
        self.cached_contact_tangential_residuals.append(tangential_residuals)

        # Clean up cache
        if len(self.cached_contact_states) > 10:
            self.cached_contact_states.pop(0)
            self.cached_contact_vars.pop(0)
            self.cached_contact_normal_residuals.pop(0)
            self.cached_contact_tangential_residuals.pop(0)

        # Store cycling window
        if cycling_window > 0:
            self.cycling_window = cycling_window
        else:
            self.cycling_window = 0

        # Store stagnating status
        self.stagnating_states = stagnating_states

        # Monitor as part of nonlinear solver statistics
        self.nonlinear_solver_statistics.cycling_window = self.cycling_window
        self.nonlinear_solver_statistics.stagnating_states = self.stagnating_states

    def before_nonlinear_loop(self):
        self.previous_states = self.compute_fracture_states(split_output=True)
        self.return_map_on = False
        super().before_nonlinear_loop()

    def after_nonlinear_iteration(self, solution_vector: np.ndarray) -> None:
        """Integrate iteration export into simulation workflow.

        Order of operations is important, super call distributes the solution
        to iterate subdictionary.

        """
        super().after_nonlinear_iteration(solution_vector)
        self.check_cycling()

    def after_nonlinear_convergence(self):
        super().after_nonlinear_convergence()
        self.reset_cycling_analysis()