from functools import partial
import numpy as np
import porepy as pp
Scalar = pp.ad.Scalar

"""Various mixins used for postprocessing of the simulations."""


class ComputeContactStates:

    def compute_contact_states(self, split_output: bool = False) -> list:
        # Compute contact states of each fracture cell using the displacement increment.
        # Returns a list where:
        # Open=0, Stick=1, Slip=2
        subdomains = self.mdg.subdomains(dim=self.nd - 1)
        u_t = self.tangential_component(subdomains) @ self.displacement_jump(subdomains)

        # # Cumulative fracture states
        utval = u_t.value(self.equation_system)
        utinc = utval-self.u_t_init
        if self.nd == 3:
            utinc_new = utinc.reshape((int(utinc.size / 2), 2))
        else:
            utinc_new = utinc
        states_cumulative = []
        tol = self.numerical.open_state_tolerance
        for vals in utinc_new:
            if np.linalg.norm(vals) > tol:
                states_cumulative.append(2)
            else:
                states_cumulative.append(1)

        # Fracture states relative to previous time step (i.e. non-cumulative)
        subdomains = self.mdg.subdomains(dim=self.nd - 1)
        u_t = self.tangential_component(subdomains) @ self.displacement_jump(subdomains)
        f_norm = pp.ad.Function(partial(pp.ad.l2_norm, self.nd - 1), "norm_function")
        tangential_basis: list[pp.ad.SparseArray] = self.basis(
            subdomains, dim=self.nd - 1  # type: ignore[call-arg]
        )
        t_n = self.normal_component(subdomains) @ self.contact_traction(subdomains)
        t_t = self.tangential_component(subdomains) @ self.contact_traction(subdomains)
        u_n = self.normal_component(subdomains) @ self.displacement_jump(subdomains)
        u_t_increment: pp.ad.Operator = pp.ad.time_increment(u_t)
        c_num_as_scalar = self.contact_mechanics_numerical_constant(subdomains)
        scalar_to_tangential = pp.ad.sum_projection_list(tangential_basis)
        # The fracture states are defined differently for IRM, as the
        # complementarity functions are regularized in that case.
        if hasattr(self, "t_t_prev"):  # IRM is used
            t_t_prev_ad = pp.ad.DenseArray(self.t_t_prev)
            tangential_sum = (
                t_t_prev_ad + (scalar_to_tangential @ c_num_as_scalar) * u_t_increment
            )
        else:   # GNM or GNM-RM is used
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
            split_states_cumulative = []
            num_cells = []
            for sd in subdomains:
                prev_num_cells = int(sum(num_cells))
                split_states.append(
                    np.array(states[prev_num_cells : prev_num_cells + sd.num_cells])
                )
                split_states_cumulative.append(
                    np.array(states_cumulative[prev_num_cells : prev_num_cells + sd.num_cells])
                )
                num_cells.append(sd.num_cells)
            return split_states, split_states_cumulative
        else:
            return states_cumulative


class CustomExporter(ComputeContactStates):
    """Export fracture contact states at every time step."""

    def __init__(self, params):
        super().__init__(params)
        self.u_t_init = 0
        self.aperture_init = 0
        self.displacement_init = 0

    def prepare_simulation(self):
        super().prepare_simulation()
        subdomains = self.mdg.subdomains()
        matrix_subdomains = self.mdg.subdomains(dim=self.nd)
        fracture_subdomains = self.mdg.subdomains(dim=self.nd - 1)
        u_t = self.tangential_component(fracture_subdomains) @ self.displacement_jump(fracture_subdomains)
        aperture = self.aperture(subdomains)
        displacement = self.displacement(matrix_subdomains)
        self.u_t_init = u_t.value(self.equation_system)
        self.aperture_init = aperture.value(self.equation_system)
        self.displacement_init = displacement.value(self.equation_system)

    def data_to_export(self):
        data = super().data_to_export()
        sds = self.mdg.subdomains()
        sds_mat = self.mdg.subdomains(dim=self.nd)
        sds_frac = self.mdg.subdomains(dim=self.nd - 1)
        states, states_cumulative = self.compute_contact_states(split_output=True)
        aperture_diff = self.aperture(sds).value(self.equation_system) - self.aperture_init
        displacement_diff = self.displacement(sds_mat).value(self.equation_system) - self.displacement_init
        cell_offsets = np.cumsum([0] + [sd.num_cells for sd in sds])
        cell_offsets_nd = np.cumsum([0] + [sd.num_cells * self.nd for sd in sds_mat])
        for i, sd in enumerate(sds_mat):
            data.append((sd, 
                         "displacement_diff", 
                        displacement_diff[cell_offsets_nd[i] : cell_offsets_nd[i + 1]]))
        for i, sd in enumerate(sds_frac):
            data.append((sd, "states", states[i]))
            data.append((sd, "states_cumulative", states_cumulative[i]))
        for i, sd in enumerate(sds):
            data.append(
                (
                    sd,
                    "aperture_diff",
                    aperture_diff[cell_offsets[i] : cell_offsets[i + 1]],
                )
            )
        return data


class ExportInjectionCell:

    """Export the cell containing the injection well."""

    def initialize_data_saving(self):
        """Initialize iteration exporter."""
        super().initialize_data_saving()
        # Setting export_constants_separately to False facilitates operations such as
        # filtering by dimension in ParaView and is done here for illustrative purposes.
        self.iteration_exporter = pp.Exporter(
            self.mdg,
            file_name=self.params["file_name"] + "_one_cell",
            folder_name=self.params["folder_name"],
            export_constants_separately=False,
        )

    def _fracture_center_cell(self, sd: pp.Grid) -> np.ndarray:
        # Compute the fracture cell that is closest to the center of the fracture
        mean_coo = np.mean(sd.cell_centers, axis=1).reshape((3, 1))
        center_cell = sd.closest_cell(mean_coo)
        return center_cell

    def center_of_first_fracture(self):
        values = []

        for sd in self.mdg.subdomains(dim=self.nd-1):
            if sd == self.mdg.subdomains(dim=self.nd-1)[0]:  # Subdomain corresponding to first fracture
                vals = np.zeros(sd.num_cells)
                center = self._fracture_center_cell(sd)
                vals[center] = 1
                values.append((sd, "onecell", vals))
            else:
                values.append((sd, "onecell", np.zeros(sd.num_cells)))
        return values
    
    def after_nonlinear_convergence(self):
        super().after_nonlinear_convergence()
        self.iteration_exporter.write_vtu(
            self.center_of_first_fracture(),
            time_dependent=True
        )