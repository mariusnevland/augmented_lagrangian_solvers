import numpy as np
import porepy as pp
from typing import Literal


class SimpleFractureNetwork:

    def set_domain(self) -> None:
        self._domain = pp.Domain(
            bounding_box={
                "xmin": 0 / self.units.m,
                "ymin": 0 / self.units.m,
                "xmax": 2000 / self.units.m,
                "ymax": 1000 / self.units.m,
            }
        )

    def set_fractures(self) -> None:
        frac_center = pp.LineFracture(1000 * np.array([[0.8, 0.3], [1.2, 0.6]]).T / self.units.m)
        frac1 = pp.LineFracture(1000 * np.array([[0.6, 0.7], [1.3, 0.4]]).T / self.units.m)
        frac2 = pp.LineFracture(1000 * np.array([[0.5, 0.2], [0.8, 0.8]]).T / self.units.m)
        self._fractures = [frac_center, frac1, frac2]

    def grid_type(self) -> Literal["simplex", "cartesian", "tensor_grid"]:
        return self.params.get("grid_type", "simplex")

    def meshing_arguments(self) -> dict:
        mesh_args = {"cell_size": 1000 * 0.07 / self.units.m}
        return mesh_args


class EvenMoreFractures:

    def set_domain(self) -> None:
        self._domain = pp.Domain(
            bounding_box={
                "xmin": 0 / self.units.m,
                "ymin": 0 / self.units.m,
                "xmax": 2000 / self.units.m,
                "ymax": 1000 / self.units.m,
            }
        )

    def set_fractures(self) -> None:
        frac_center = pp.LineFracture(1000 * np.array([[0.8, 0.3], [1.2, 0.6]]).T / self.units.m)
        frac1 = pp.LineFracture(1000 * np.array([[0.6, 0.7], [1.3, 0.4]]).T / self.units.m)
        frac2 = pp.LineFracture(1000 * np.array([[0.6, 0.25], [0.8, 0.8]]).T / self.units.m)
        frac3 = pp.LineFracture(1000 * np.array([[0.25, 0.5], [0.5, 0.65]]).T / self.units.m)
        frac4 = pp.LineFracture(1000 * np.array([[0.3, 0.1], [0.6, 0.7]]).T / self.units.m)
        frac5 = pp.LineFracture(1000 * np.array([[1.0, 0.7], [1.4, 0.7]]).T / self.units.m)
        frac6 = pp.LineFracture(1000 * np.array([[1.05, 0.35], [1.6, 0.8]]).T / self.units.m)
        frac7 = pp.LineFracture(1000 * np.array([[0.3, 0.7], [0.55, 0.35]]).T / self.units.m)
        frac8 = pp.LineFracture(1000 * np.array([[1.4, 0.9], [1.8, 0.5]]).T / self.units.m)
        frac9 = pp.LineFracture(1000 * np.array([[1.4, 0.1], [1.7, 0.4]]).T / self.units.m)
        frac10 = pp.LineFracture(1000 * np.array([[1.2, 0.3], [1.8, 0.1]]).T / self.units.m)
        frac11 = pp.LineFracture(1000 * np.array([[0.9, 0.45], [0.9, 0.15]]).T / self.units.m)
        frac12 = pp.LineFracture(1000 * np.array([[0.7, 0.1], [1.0, 0.25]]).T / self.units.m)
        frac13 = pp.LineFracture(1000 * np.array([[0.75, 0.4], [1.0, 0.7]]).T / self.units.m)
        frac14 = pp.LineFracture(1000 * np.array([[1.27, 0.6], [1.1, 0.85]]).T / self.units.m)
        frac15 = pp.LineFracture(1000 * np.array([[1.5, 0.5], [1.75, 0.2]]).T / self.units.m)
        frac16 = pp.LineFracture(1000 * np.array([[0.7, 0.38], [0.83, 0.1]]).T / self.units.m)
        frac17 = pp.LineFracture(1000 * np.array([[0.5, 0.83], [0.85, 0.65]]).T / self.units.m)
        frac18 = pp.LineFracture(1000 * np.array([[1.1, 0.15], [1.4, 0.4]]).T / self.units.m)
        self._fractures = [frac_center, frac1, frac2, frac3, frac4, frac5, frac6, frac7, frac8, frac9,
                           frac10, frac11, frac12, frac13, frac14, frac15, frac16, frac17, frac18]

    def grid_type(self) -> Literal["simplex", "cartesian", "tensor_grid"]:
        return self.params.get("grid_type", "simplex")

    def meshing_arguments(self) -> dict:
        mesh_args = {"cell_size": 1000 * 0.07 / self.units.m}
        return mesh_args


class AnisotropicStressBC:
    def bc_type_mechanics(self, sd: pp.Grid) -> pp.BoundaryConditionVectorial:
        all_bf, east, west, north, south, _, _ = self.domain_boundary_sides(sd)
        bc = pp.BoundaryConditionVectorial(sd, all_bf, "dir")
        bc.internal_to_dirichlet(sd)
        # Neumann conditions on east, north and west faces.
        bc.is_dir[:, east + north + west] = False
        bc.is_neu[:, east + north + west] = True
        return bc

    def bc_values_stress(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        bounds = self.domain_boundary_sides(boundary_grid)
        values = np.zeros((self.nd, boundary_grid.num_cells))
        # Lithostatic pressure at 2 km is about 50MPa.
        lith = 5e7
        values[0, bounds.north] = self.units.convert_units(0, "Pa") * boundary_grid.cell_volumes[bounds.north]
        values[1, bounds.north] = self.units.convert_units(-lith, "Pa") * boundary_grid.cell_volumes[bounds.north]
        values[1, bounds.east] = self.units.convert_units(0, "Pa") * boundary_grid.cell_volumes[bounds.east]
        values[0, bounds.east] = self.units.convert_units(-3/4*lith, "Pa") * boundary_grid.cell_volumes[bounds.east]
        values[1, bounds.west] = self.units.convert_units(0, "Pa") * boundary_grid.cell_volumes[bounds.west]
        values[0, bounds.west] = self.units.convert_units(3/4*lith, "Pa") * boundary_grid.cell_volumes[bounds.west]
        values = values.ravel("F")
        return values


class ConstantPressureBC:
    # Constant pressure of 20MPa on all boundaries.
    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        all_bf, east, west, north, south, _, _ = self.domain_boundary_sides(sd)
        # Dirichlet (pressure) conditions west and east, Neumann (flux) conditions
        # north and south.
        return pp.BoundaryCondition(sd, east + west + north + south, "dir")

    def bc_values_pressure(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        bounds = self.domain_boundary_sides(boundary_grid)
        # Hydrostatic pressure at 2 km is about 20 MPa.
        values = self.units.convert_units(2e7, "Pa") * np.ones(boundary_grid.num_cells)
        return values
    

class ConstrainedPressureEquaton:
 
     def mass_balance_equation(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
         eq = self.pressure(subdomains) - pp.ad.Scalar(self.units.convert_units(2e7, "Pa"))
         eq.set_name("new mass balance")
         return eq
    

class PressureConstraintWell:

    def update_time_dependent_ad_arrays(self) -> None:
        """Set current injection pressure."""
        super().update_time_dependent_ad_arrays()
 
        # Update injection pressure
        current_injection_pressure = self.units.convert_units(2e7, "Pa") + self.units.convert_units(1.5*1e7, "Pa")
        for sd in self.mdg.subdomains(return_data=False):
            pp.set_solution_values(
                name="current_injection_pressure",
                values=np.array([current_injection_pressure]),
                data=self.mdg.subdomain_data(sd),
                iterate_index=0,
            )
 
    def _fracture_center_cell(self, sd: pp.Grid) -> np.ndarray:
        # Compute the fracture cell that is closest to the center of the fracture
        mean_coo = np.mean(sd.cell_centers, axis=1).reshape((3, 1))
        center_cell = sd.closest_cell(mean_coo)
        return center_cell
 
    def mass_balance_equation(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        std_eq = super().mass_balance_equation(subdomains)
 
        # Need to embedd in full domain
        sd_indicator = [np.zeros(sd.num_cells) for sd in subdomains]
 
        # Pick the only subdomain
        fracture_sds = [sd for sd in subdomains if sd.dim == self.nd - 1]
 
        if len(fracture_sds) == 0:
            return std_eq
 
        # Pick a single fracture
        well_sd = fracture_sds[0]
 
        for i, sd in enumerate(subdomains):
            if sd == well_sd:
 
                well_loc_ind = self._fracture_center_cell(sd)
                sd_indicator[i][well_loc_ind] = 1
 
        # Characteristic functions
        indicator = np.concatenate(sd_indicator)
        reverse_indicator = 1 - indicator
 
        current_injection_pressure = pp.ad.TimeDependentDenseArray(
            "current_injection_pressure", [self.mdg.subdomains()[0]]
        )
        constrained_eq = self.pressure(subdomains) - current_injection_pressure
 
        eq_with_pressure_constraint = (
            pp.ad.DenseArray(reverse_indicator) * std_eq
            + pp.ad.DenseArray(indicator) * constrained_eq
        )
        eq_with_pressure_constraint.set_name(
            "mass_balance_equation_with_constrained_pressure"
        )
 
        return eq_with_pressure_constraint