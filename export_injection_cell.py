import numpy as np
import porepy as pp
# Make one cell a different color from the other cells in ParaView.

class ExportInjectionCell:

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


class ExportInjectionCellGrid:

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

    def _cell_within_well(self, sd: pp.Grid) -> np.ndarray:
        # Compute the fracture cell that is closest to the center of the fracture
        cell_within_well = []
        for i, center in enumerate(sd.cell_centers[0]):
            if center > 946 and center < 994:
                cell_within_well.append(i)
                print(center)
        return np.array(cell_within_well)

    def center_of_first_fracture(self):
        values = []

        for sd in self.mdg.subdomains(dim=self.nd-1):
            if sd == self.mdg.subdomains(dim=self.nd-1)[0]:  # Subdomain corresponding to first fracture
                vals = np.zeros(sd.num_cells)
                well_cell = self._cell_within_well(sd)
                vals[well_cell] = 1
                values.append((sd, "wellcell", vals))
            else:
                values.append((sd, "wellcell", np.zeros(sd.num_cells)))
        return values
    
    def after_nonlinear_convergence(self):
        super().after_nonlinear_convergence()
        self.iteration_exporter.write_vtu(
            self.center_of_first_fracture(),
            time_dependent=True
        )