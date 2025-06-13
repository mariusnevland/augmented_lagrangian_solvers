import numpy as np
import porepy as pp
import copy
from model_setup_example_1 import *
from run_and_report_single import *
from parameters import *
from cubic_normal_permeability import *
from export_injection_cell import *
from convergence_metrics import *
from export_iterations import *
from contact_mechanics_mixins import *
from heatmap import *

# grid_sizes = [500 * 0.07, 300 * 0.07, 175 * 0.07, 125 * 0.07]
grid_sizes = [125 * 0.07]
injection_pressures = [0.1 * 1e7]
solvers = ["NewtonReturnMap"]
itr_list = []

for size in grid_sizes:

    class Grid:

        def meshing_arguments(self) -> dict:
            mesh_args = {"cell_size": size / self.units.m}
            return mesh_args
        
    class SimpleInjectionInit(Grid,
                          MoreFocusedFractures,
                          AnisotropicStressBC,
                          ConstantPressureBC,
                          ConstrainedPressureEquaton,
                          LebesgueConvergenceMetrics,
                          AlternativeTangentialEquation,
                          ContactMechanicsConstant,
                          DimensionalContactTraction,
                          NormalPermeabilityFromSecondary,
                          pp.constitutive_laws.CubicLawPermeability,
                          pp.poromechanics.Poromechanics):
        pass

    params = copy.deepcopy(params_initialization)
    model = SimpleInjectionInit(params)
    pp.run_time_dependent_model(model, params)
    vals = model.equation_system.get_variable_values(iterate_index=0)

    class InitialCondition:
        def initial_condition(self) -> None:
            super().initial_condition()
            self.equation_system.set_variable_values(
                vals,
                time_step_index=0,
                iterate_index=0,
            )

    class SimpleInjection(InitialCondition,
                      Grid,
                      MoreFocusedFractures,
                      PressureConstraintWellGrid,
                      AnisotropicStressBC,
                      ConstantPressureBC,
                      LebesgueConvergenceMetrics,
                      AlternativeTangentialEquation,
                      DimensionalContactTraction,
                      NormalPermeabilityFromSecondary,
                      pp.constitutive_laws.CubicLawPermeability,
                      pp.poromechanics.Poromechanics):
        pass

    for pressure in injection_pressures:
        for solver in solvers:
            params = copy.deepcopy(params_injection_2D)
            params["injection_overpressure"] = pressure
            itr_solver = run_and_report_single(Model=SimpleInjection, params=params, c_value=1e-1, solver=solver)
            itr_list.append(itr_solver)
            print(itr_solver)
# itr_list = np.array(itr_list).reshape((4, 4)).T
# xticks = ["4879", "12130", "33790", "64028"]
# yticks = ["Newton, well pressure 21MPa", "NRM, well pressure 21MPa",
#           "Newton, well pressure 25MPa", "NRM, well pressure 25MPa"]
# heatmap(data=itr_list, vmin=1, vmax=200, xticks=xticks, yticks=yticks,
#             xlabel="Total number of cells", file_name="fig7")