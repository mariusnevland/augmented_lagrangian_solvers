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
from numpy import radians as rad

class SimpleInjectionInit(VerticalHorizontalNetwork,
                          AnisotropicStressBC,
                          ConstantPressureBC,
                          ConstrainedPressureEquaton,
                          LebesgueConvergenceMetrics,
                          AlternativeTangentialEquation,
                          ContactMechanicsConstant,
                          DimensionalContactTraction,
                          # IterationExporting,
                          NormalPermeabilityFromSecondary,
                          pp.constitutive_laws.CubicLawPermeability,
                          pp.poromechanics.Poromechanics):
    pass

c_values = [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2]
solvers = ["Newton", "NewtonReturnMap"]
dilation_angles = [rad(0), rad(2), rad(4), rad(6), rad(8), rad(10)]

itr_list = []
for dil in dilation_angles:
    params_init = copy.deepcopy(params_initialization)
    params_init["material_constants"]["solid"].dilation_angle = dil
    model_init = SimpleInjectionInit(params_init)
    try:
        pp.run_time_dependent_model(model_init, params_init)
    except ValueError or RuntimeError:
        print("fail")
    vals = model_init.equation_system.get_variable_values(iterate_index=0)

    class InitialCondition:
        def initial_condition(self) -> None:
            super().initial_condition()
            self.equation_system.set_variable_values(
            vals,
            time_step_index=0,
            iterate_index=0,
        )
            
    class SimpleInjection(InitialCondition,
                      VerticalHorizontalNetwork,
                      PressureConstraintWell,
                      AnisotropicStressBC,
                      ConstantPressureBC,
                      LebesgueConvergenceMetrics,
                      AlternativeTangentialEquation,
                      DimensionalContactTraction,
                      # IterationExporting,
                      NormalPermeabilityFromSecondary,
                      pp.constitutive_laws.CubicLawPermeability,
                      pp.poromechanics.Poromechanics):
        pass

    for solver in solvers:
        params = copy.deepcopy(params_injection_2D)
        params["material_constants"]["solid"].dilation_angle = dil
        itr_solver = run_and_report_single(Model=SimpleInjection, params=params, c_value=1e-2, solver=solver)
        itr_list.append(itr_solver)
itr_list = np.array(itr_list).reshape((len(dilation_angles),len(solvers)))
print(itr_list)