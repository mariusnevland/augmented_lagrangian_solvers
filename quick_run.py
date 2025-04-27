import porepy as pp
from typing import Literal
from parameters import *


class NoFractures:

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
        self._fractures = []

    def grid_type(self) -> Literal["simplex", "cartesian", "tensor_grid"]:
        return self.params.get("grid_type", "simplex")

    def meshing_arguments(self) -> dict:
        mesh_args = {"cell_size": 1000 * 0.07 / self.units.m}
        return mesh_args
    

class Simulation(NoFractures,
                 pp.momentum_balance.MomentumBalance):
    pass

model = Simulation(params_outline)
pp.run_time_dependent_model(model, params_outline)