import numpy as np
import porepy as pp
import copy
from model_setup_example_1 import *
from run_and_report_single import *
from parameters import *
from cubic_normal_permeability import *
from convergence_metrics import *
from export_iterations import *
from contact_mechanics_mixins import *

grid_sizes = [1000 * 0.07, 500 * 0.07, 250 * 0.07]
itr_list = []

for size in grid_sizes:

    class Grid:

        def meshing_arguments(self) -> dict:
            mesh_args = {"cell_size": size / self.units.m}
            return mesh_args
        
    class SimpleInjectionInit(Grid,
                          VerticalHorizontalNetwork,
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
                      VerticalHorizontalNetwork,
                      PressureConstraintWell,
                      AnisotropicStressBC,
                      ConstantPressureBC,
                      LebesgueConvergenceMetrics,
                      AlternativeTangentialEquation,
                      DimensionalContactTraction,
                      NormalPermeabilityFromSecondary,
                      pp.constitutive_laws.CubicLawPermeability,
                      pp.poromechanics.Poromechanics):
        pass

    params = copy.deepcopy(params_injection_2D)
    itr_solver = run_and_report_single(Model=SimpleInjection, params=params, c_value=1e-2, solver="ClassicalReturnMap")
    itr_list.append(itr_solver)
print(itr_list)