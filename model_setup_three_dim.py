import numpy as np
import porepy as pp
from typing import Literal


class EllipticFractureNetwork:

    def set_domain(self) -> None:
        self._domain = \
            pp.Domain(bounding_box=
                      {'xmin': 0 / self.units.m, 'xmax': 2000 / self.units.m,
                       'ymin': 0 / self.units.m, 'ymax': 2000 / self.units.m,
                       'zmin': 0 / self.units.m, 'zmax': 2000 / self.units.m})

    def set_fractures(self) -> None:
        f_1 = pp.create_elliptic_fracture(
                np.array([800, 800, 1200]),
                600,
                300,
                0.5,
                np.pi / 4,
                np.pi / 4,
            )
        f_2 = pp.create_elliptic_fracture(
                np.array([800, 800, 1000]),
                600,
                300,
                -0.5 ,
                -np.pi / 4,
                -np.pi / 4,
            )
        f_3 = pp.create_elliptic_fracture(
                np.array([400, 1500, 700]),
                500,
                300,
                0.5 ,
                -np.pi / 3,
                np.pi / 4,
            )
        f_4 = pp.create_elliptic_fracture(
                np.array([1500, 1200, 500]),
                400,
                500,
                0.2 ,
                np.pi / 2,
                np.pi / 4,
            )
        f_5 = pp.create_elliptic_fracture(
                np.array([1100, 1100, 1600]),
                300,
                400,
                -np.pi / 2 ,
                0,
                np.pi / 3,
            )
        self._fractures = [f_1, f_2, f_3, f_4, f_5]

    def grid_type(self) -> Literal["simplex", "cartesian", "tensor_grid"]:
        return self.params.get("grid_type", "simplex")

    def meshing_arguments(self) -> dict:
        mesh_args = {"cell_size": 1500 * 0.3 / self.units.m}
        return mesh_args
    

class LithoStaticTraction3D:

    # The z-component of the litostatic traction follows the formula p=rho*g*z.
    # The x- and y-components are proportional to the z-component.

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
        scale_x, scale_y, scale_z = 3/4, 5/4, 1
        values[0, east] = -scale_x * lithostatic[east] * cell_volumes[east]
        values[2, top] = -scale_z * lithostatic[top] * cell_volumes[top]
        values[0, west] = scale_x * lithostatic[west] * cell_volumes[west]
        values[1, north] = -scale_y * lithostatic[north] * cell_volumes[north]
        values[1, south] = scale_y * lithostatic[south] * cell_volumes[south]
        values = values.ravel("F")
        return values
    

class HydrostaticPressureGradient3D:

    # Pressure gradient directly based on the hydrostatic pressure, i.e 
    # the pressure will linearly increase from top to bottom, according to
    # p=rho*g*z.

    # Dirichlet (pressure) conditions on all boundaries. That is the default
    # option for the bc_type methods, so they do not need to be overwritten.

    def _depth(self, coords) -> np.ndarray:
        # We assume that the bottom of our domain (which has z-coordinate zero)
        # is at 3km depth.
        return 3000 - coords

    def hydrostatic_pressure(self, depth):
        rho = self.params["material_constants"]["fluid"].constants_in_SI["density"]
        return self.units.convert_units(rho * pp.GRAVITY_ACCELERATION * depth, "Pa")

    def bc_values_pressure(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        values = np.zeros(boundary_grid.num_cells)
        # Find the depth of the cell centers, i.e. their z-coordinates.
        depth = self._depth(boundary_grid.cell_centers[2,:])
        all_bf, *_ = self.domain_boundary_sides(boundary_grid)
        values[all_bf] = self.hydrostatic_pressure(depth)
        return values
    