import numpy as np
import porepy as pp
from typing import Literal, Callable


class EllipticFractureNetwork:

    def set_domain(self) -> None:
        self._domain = \
            pp.Domain(bounding_box=
                      {'xmin': 0 / self.units.m, 'xmax': 2000 / self.units.m,
                       'ymin': 0 / self.units.m, 'ymax': 2000 / self.units.m,
                       'zmin': 0 / self.units.m, 'zmax': 2000 / self.units.m})

    def set_fractures(self) -> None:
        f_1 = pp.create_elliptic_fracture(
                np.array([800, 800, 900]),
                600,
                300,
                0.5,
                np.pi / 3,
                np.pi / 4,
            )
        f_2 = pp.create_elliptic_fracture(
                np.array([500, 700, 1100]),
                600,
                300,
                -0.5 ,
                -np.pi / 4,
                -np.pi / 4,
            )
        f_3 = pp.create_elliptic_fracture(
                np.array([800, 1000, 1200]),
                400,
                500,
                0.2 ,
                0,
                np.pi / 2,
            )
        self._fractures = [f_1, f_2, f_3]

    def grid_type(self) -> Literal["simplex", "cartesian", "tensor_grid"]:
        return self.params.get("grid_type", "simplex")

    def meshing_arguments(self) -> dict:
        mesh_args = {"cell_size": 700 * 0.3 / self.units.m,
                     "cell_size_fracture": 300 * 0.3 / self.units.m}
        return mesh_args
    

class LithoStaticTraction3D:

    # The z-component of the litostatic traction follows the formula p=rho*g*z.
    # The x- and y-components are proportional to the z-component.

    # The bottom boundary is kept fixed, while compressive tractions are imposed
    # on the other boundaries.

    def _depth(self, coords) -> np.ndarray:
        # We assume that the bottom of our domain (which has z-coordinate zero)
        # is at 3km depth.
        return 3000 - coords
    
    def lithostatic_pressure(self, depth):
        rho = self.params["material_constants"]["solid"].constants_in_SI["density"]
        return self.units.convert_units(rho * pp.GRAVITY_ACCELERATION * depth, "Pa")
    
    def bc_type_mechanics(self, sd: pp.Grid) -> pp.BoundaryConditionVectorial:
        all_bf, east, west, north, south, top, bottom = \
            self.domain_boundary_sides(sd)
        bc = pp.BoundaryConditionVectorial(sd, all_bf, "dir")
        bc.internal_to_dirichlet(sd)
        bc.is_dir[:, east + top + west + north + south] = False
        bc.is_neu[:, east + top + west + north + south] = True
        return bc

    def bc_values_stress(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        all_bf, east, west, north, south, top, bottom = self.domain_boundary_sides(boundary_grid)
        values = np.zeros((self.nd, boundary_grid.num_cells))
        cell_volumes = boundary_grid.cell_volumes
        depth = self._depth(boundary_grid.cell_centers[2,:])
        lithostatic = self.lithostatic_pressure(depth)
        scale_x, scale_y, scale_z = 0.6, 0.6, 1
        values[0, east] = -scale_x * lithostatic[east] * cell_volumes[east]
        values[2, top] = -scale_z * lithostatic[top] * cell_volumes[top]
        values[0, west] = scale_x * lithostatic[west] * cell_volumes[west]
        values[1, north] = -scale_y * lithostatic[north] * cell_volumes[north]
        values[1, south] = scale_y * lithostatic[south] * cell_volumes[south]
        values = values.ravel("F")
        return values
    

class ConstrainedPressureEquaton3D:

     def mass_balance_equation(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
         eq = self.pressure(subdomains) - pp.ad.Scalar(self.units.convert_units(2e7, "Pa"))
         eq.set_name("new mass balance")
         return eq
    

class HydrostaticPressure:
    """Utility class to compute (generalized) hydrostatic pressure."""

    fluid: pp.FluidComponent

    def hydrostatic_pressure(self, sd: pp.Grid) -> np.ndarray:
        p_atm = 0
        gravity = self.units.convert_units(pp.GRAVITY_ACCELERATION, "m*s^-2")
        rho = self.params["material_constants"]["fluid"].constants_in_SI["density"]
        rho_scaled = self.units.convert_units(rho, "Pa")
        rho_g = rho_scaled * gravity
        # We assume the top of the domain is at 1km depth.
        z = self.units.convert_units(3000, "m") - sd.cell_centers[self.nd - 1] 
        pressure = p_atm + rho_g * z
        return pressure

    def update_time_dependent_ad_arrays(self) -> None:
        """Set hydrostatic pressure for current gravity."""
        super().update_time_dependent_ad_arrays()

        # Update injection pressure
        for sd in self.mdg.subdomains(return_data=False):
            hydrostatic_pressure = self.hydrostatic_pressure(sd)
            pp.set_solution_values(
                name="hydrostatic_pressure",
                values=np.array(hydrostatic_pressure),
                data=self.mdg.subdomain_data(sd),
                iterate_index=0,
            )


class HydrostaticPressureBC(HydrostaticPressure):

    # Pressure gradient directly based on the hydrostatic pressure, i.e 
    # the pressure will linearly increase from top to bottom, according to
    # p=rho*g*z.

    # Dirichlet (pressure) conditions on all boundaries.

    domain_boundary_sides: Callable[[pp.Grid | pp.BoundaryGrid], pp.domain.DomainSides]

    def _fluid_pressure_boundary_faces(self, sd: pp.Grid) -> np.ndarray:
        """Auxiliary method to identify all Dirichlet/pressure boundaries."""
        domain_sides = self.domain_boundary_sides(sd)
        return domain_sides.all_bf

    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        return pp.BoundaryCondition(sd, self._fluid_pressure_boundary_faces(sd), "dir")

    def bc_type_fluid_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        return pp.BoundaryCondition(sd, self._fluid_pressure_boundary_faces(sd), "dir")

    def bc_values_pressure(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        pressure = np.zeros(boundary_grid.num_cells)

        # Apply hydrostatic pressure on all sides of the domain.
        if boundary_grid.dim == self.nd - 1:
            sides = self.domain_boundary_sides(boundary_grid)
            pressure[sides.all_bf] = self.hydrostatic_pressure(boundary_grid)[
                sides.all_bf
            ]

        return pressure
    

class HydroStaticPressureInitialization(HydrostaticPressure):

    # Set the pressure equal to the hydrostatic pressure, to be used for initialization.

    def mass_balance_equation(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Overwrites mass balance equation to set the pressure equal to the hydrostatic pressure."""
        hydrostatic_pressure = pp.ad.TimeDependentDenseArray(
            "hydrostatic_pressure", subdomains
        )
        constrained_eq = self.pressure(subdomains) - hydrostatic_pressure
        constrained_eq.set_name("mass_balance_equation")
        return constrained_eq
    

class PressureConstraintWell3D:
    """Pressurize specific fractures in their center, representing an injection well."""

    def _fracture_center_cell(self, sd: pp.Grid) -> np.ndarray:
        # Compute the fracture cell that is closest to the center of the fracture
        mean_coo = np.mean(sd.cell_centers, axis=1).reshape((3, 1))
        center_cell = sd.closest_cell(mean_coo)
        return center_cell

    def update_time_dependent_ad_arrays(self) -> None:
        """Set current injection pressure."""
        super().update_time_dependent_ad_arrays()

        # Update injection pressure
        injection_index = self.time_manager._scheduled_idx
        injection_schedule = [1, 1, 2, 2, 3, 3]
        injection_overpressure = self.params.get("injection_overpressure", 0)
        current_injection_overpressure = injection_schedule[injection_index-1] * injection_overpressure
        for sd in self.mdg.subdomains(return_data=False):
            pp.set_solution_values(
                name="current_injection_overpressure",
                values=np.array(
                    [self.units.convert_units(current_injection_overpressure, "Pa")]
                ),
                data=self.mdg.subdomain_data(sd),
                iterate_index=0,
            )

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

        current_injection_overpressure = pp.ad.TimeDependentDenseArray(
            "current_injection_overpressure", [self.mdg.subdomains()[0]]
        )
        hydrostatic_pressure = pp.ad.TimeDependentDenseArray(
            "hydrostatic_pressure", subdomains
        )
        constrained_eq = (
            self.pressure(subdomains)
            - current_injection_overpressure
            - hydrostatic_pressure
        )

        eq_with_pressure_constraint = (
            pp.ad.DenseArray(reverse_indicator) * std_eq
            + pp.ad.DenseArray(indicator) * constrained_eq
        )
        eq_with_pressure_constraint.set_name(
            "mass_balance_equation"
        )

        return eq_with_pressure_constraint
    