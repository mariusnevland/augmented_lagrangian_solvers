from functools import partial
import numpy as np
import porepy as pp
Scalar = pp.ad.Scalar


class IterationExporting:
    @property
    def iterate_indices(self):
        """Force storing all previous iterates."""
        return np.array([0, 1])

    def initialize_data_saving(self):
        """Initialize iteration exporter."""
        super().initialize_data_saving()
        # Setting export_constants_separately to False facilitates operations such as
        # filtering by dimension in ParaView and is done here for illustrative purposes.
        self.iteration_exporter = pp.Exporter(
            self.mdg,
            file_name=self.params["file_name"] + "_iterations",
            folder_name=self.params["folder_name"],
            export_constants_separately=False,
        )

    def data_to_export_iteration(self):
        """Returns data for iteration exporting.

        Returns:
            Any type compatible with data argument of pp.Exporter().write_vtu().

        """
        # The following is a slightly modified copy of the method
        # data_to_export() from DataSavingMixin.
        data = []
        variables = self.equation_system.variables
        for var in variables:
            # Note that we use iterate_index=0 to get the current solution, whereas
            # the regular exporter uses time_step_index=0.
            scaled_values = self.equation_system.get_variable_values(
                variables=[var], iterate_index=0
            )
            units = var.tags["si_units"]
            values = self.units.convert_units(scaled_values, units, to_si=True)
            data.append((var.domain, var.name, values))

            # Append increments if available
            try:
                prev_scaled_values = self.equation_system.get_variable_values(
                    variables=[var], iterate_index=1
                )
                inc_values = self.units.convert_units(
                    scaled_values - prev_scaled_values, units, to_si=True
                )
            except:
                inc_values = self.units.convert_units(
                    scaled_values - scaled_values, units, to_si=True
                )
            data.append((var.domain, var.name + "_inc", inc_values))

        # Add contact states
        states = self.compute_fracture_states(split_output=True)
        # print(states)
        try:
            prev_states = self.prev_states.copy()
        except:
            prev_states = states.copy()
        # data.append((self.mdg.subdomains(dim=self.nd-1), "states", states))
        for i, sd in enumerate(self.mdg.subdomains(dim=self.nd - 1)):
            data.append((sd, "states", states[i]))
            data.append((sd, "prev states", prev_states[i]))
        for sd in self.mdg.subdomains(dim=self.nd - 1):
            normal_traction = self.normal_component([sd]) @ self.contact_traction([sd])
            tangential_traction = self.tangential_component([sd]) @ self.contact_traction([sd])
            data.append((sd, "tangential_traction", tangential_traction.value(self.equation_system)))
            data.append((sd, "normal_traction", normal_traction.value(self.equation_system)))
        # Cache contact states
        self.prev_state = states.copy()

        return data
    
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

    def save_data_iteration(self):
        """Export current solution to vtu files.

        This method is typically called by after_nonlinear_iteration.

        Having a separate exporter for iterations avoids distinguishing between iterations
        and time steps in the regular exporter's history (used for export_pvd).

        """
        # To make sure the nonlinear iteration index does not interfere with the
        # time part, we multiply the latter by the next power of ten above the
        # maximum number of nonlinear iterations. Default value set to 10 in
        # accordance with the default value used in NewtonSolver
        n = self.params.get("max_iterations", 10)
        p = round(np.log10(n))
        r = 10**p
        if r <= n:
            r = 10 ** (p + 1)
        self.iteration_exporter.write_vtu(
            self.data_to_export_iteration(),
            time_dependent=True,
            time_step=self.nonlinear_solver_statistics.num_iteration + r * self.time_manager.time_index,
        )

    def after_nonlinear_iteration(self, solution_vector: np.ndarray) -> None:
        """Integrate iteration export into simulation workflow.

        Order of operations is important, super call distributes the solution to
        iterate subdictionary.

        """
        super().after_nonlinear_iteration(solution_vector)
        self.save_data_iteration()
        self.iteration_exporter.write_pvd()
        print()  # force progressbar to output.