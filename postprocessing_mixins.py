from functools import partial
import numpy as np
import porepy as pp
Scalar = pp.ad.Scalar

"""Various mixins used for postprocessing of the simulations."""


class ComputeContactStates:

    def compute_contact_states(self, split_output: bool = False) -> list:
        # Compute contact states of each fracture cell using the displacement increment.
        # Returns a list where:
        # Stick=1, Slip=2
        subdomains = self.mdg.subdomains(dim=self.nd - 1)
        u_t = self.tangential_component(subdomains) @ self.displacement_jump(subdomains)
        # # Cumulative fracture states
        utval = u_t.value(self.equation_system)
        utinc = utval-self.u_t_init
        states = []
        tol = self.numerical.open_state_tolerance
        for vals in utinc:
            if np.linalg.norm(vals) > tol:
                states.append(2)
            else:
                states.append(1)

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


class CustomExporter(ComputeContactStates):
    """Export fracture contact states at every time step."""

    def __init__(self, params):
        super().__init__(params)
        self.u_t_init = 0
        self.aperture_init = 0
        self.displacement_init = 0

    def prepare_simulation(self):
        super().prepare_simulation()
        matrix_subdomains = self.mdg.subdomains(dim=self.nd)
        fracture_subdomains = self.mdg.subdomains(dim=self.nd - 1)
        u_t = self.tangential_component(fracture_subdomains) @ self.displacement_jump(fracture_subdomains)
        aperture = self.aperture(fracture_subdomains)
        displacement = self.displacement(matrix_subdomains)
        self.u_t_init = u_t.value(self.equation_system)
        self.aperture_init = aperture.value(self.equation_system)
        self.displacement_init = displacement.value(self.equation_system)

    def data_to_export(self):
        data = super().data_to_export()
        sds_mat = self.mdg.subdomains(dim=self.nd)
        sds_frac = self.mdg.subdomains(dim=self.nd - 1)
        states = self.compute_contact_states(split_output=True)
        aperture_diff = self.aperture(sds_frac).value(self.equation_system) - self.aperture_init
        displacement_diff = self.displacement(sds_mat).value(self.equation_system) - self.displacement_init
        cell_offsets = np.cumsum([0] + [sd.num_cells for sd in sds_frac])
        cell_offsets_nd = np.cumsum([0] + [sd.num_cells * self.nd for sd in sds_mat])
        for i, sd in enumerate(sds_mat):
            data.append((sd, 
                         "displacement_diff", 
                        displacement_diff[cell_offsets_nd[i] : cell_offsets_nd[i + 1]]))
        for i, sd in enumerate(sds_frac):
            data.append((sd, "states", states[i]))
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