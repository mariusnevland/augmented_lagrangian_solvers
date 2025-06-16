import porepy as pp
from typing import Literal
from parameters import *
from model_setup_example_1 import *
from model_setup_three_dim import *
from export_injection_cell import *

size = 125 * 0.07

class Grid:

    def meshing_arguments(self) -> dict:
        mesh_args = {"cell_size": size / self.units.m}
        return mesh_args
    
class NoFractures2D(MoreFocusedFractures):

    def set_fractures(self) -> None:
        self._fractures = []
   

class NoFractures3D:

    def set_domain(self) -> None:
        self._domain = \
            pp.Domain(bounding_box=
                      {'xmin': 0 / self.units.m, 'xmax': 2000 / self.units.m,
                       'ymin': 0 / self.units.m, 'ymax': 2000 / self.units.m,
                       'zmin': 0 / self.units.m, 'zmax': 2000 / self.units.m})

    def set_fractures(self) -> None:
        self._fractures = []

    def grid_type(self) -> Literal["simplex", "cartesian", "tensor_grid"]:
        return self.params.get("grid_type", "simplex")

    def meshing_arguments(self) -> dict:
        mesh_args = {"cell_size": 500 * 0.3 / self.units.m}
        return mesh_args

class Simulation(EllipticFractureNetwork,
                 ExportInjectionCell,
                 pp.momentum_balance.MomentumBalance):
    pass

model = Simulation(params_export_injection_cell)
pp.run_time_dependent_model(model, params_export_injection_cell)