import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import porepy as pp
import copy
from model_setup_two_dim import *
from plot_utils import *
from parameters import *
from model_setup_common import *
from postprocessing_mixins import *
from porepy.applications.test_utils.models import add_mixin

# Runscript for producing figure ? in the article.

class SimpleInjectionInit(FractureNetwork2D,
                          AnisotropicStressBC,
                          ConstantPressureBC,
                          CustomPressureStress,
                          ConstrainedPressureEquation,
                          LebesgueConvergenceMetrics,
                          AlternativeTangentialEquation,
                          ContactMechanicsConstant,
                          DimensionalContactTraction,
                          NormalPermeabilityFromSecondary,
                          pp.constitutive_laws.CubicLawPermeability,
                          pp.poromechanics.Poromechanics):
    pass

class SimpleInjection(FractureNetwork2D,
                      PressureConstraintWell,
                      AnisotropicStressBC,
                      ConstantPressureBC,
                      CustomPressureStress,
                      LebesgueConvergenceMetrics,
                      AlternativeTangentialEquation,
                      DimensionalContactTraction,
                      NormalPermeabilityFromSecondary,
                      pp.constitutive_laws.CubicLawPermeability,
                      pp.poromechanics.Poromechanics):
    pass


c_values = [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4]
solvers = ["GNM", "GNM-RM", "IRM"]
itr_list = []
nonlinearities = ["no_aperture", "no_cubic_law", "full_model"]
for nonlin in nonlinearities:
    for solver in solvers:
        for c in c_values:
            params_init = copy.deepcopy(params_initialization)
            if nonlin == "no_aperture":
                model_class_init = add_mixin(ConstantAperture, SimpleInjectionInit)
            elif nonlin == "no_cubic_law":
                model_class_init = add_mixin(ConstantCubicLawPermeability, SimpleInjectionInit)
            elif nonlin == "full_model":
                model_class_init = SimpleInjectionInit
            model_init = model_class_init(params_init)
            pp.run_time_dependent_model(model_init, params_init)
            vals = model_init.equation_system.get_variable_values(iterate_index=0)
            class InitialCondition:
                def initial_condition(self) -> None:
                    super().initial_condition()
                    self.equation_system.set_variable_values(
                        vals,
                        time_step_index=0,
                        iterate_index=0,
                    )
            if nonlin == "no_aperture":
                model_class = add_mixin(InitialCondition, add_mixin(ConstantAperture, SimpleInjection))
            elif nonlin == "no_cubic_law":
                model_class = add_mixin(InitialCondition, add_mixin(ConstantCubicLawPermeability, SimpleInjection))
            elif nonlin == "full_model":
                model_class = add_mixin(InitialCondition, SimpleInjection)      
            params = copy.deepcopy(params_injection_2D)
            params["injection_overpressure"] = 0.1 * 1e7
            itr_solver = run_and_report_single(Model=model_class, params=params, c_value=c, solver=solver)
            itr_list.append(itr_solver)
    itr_list = np.array(itr_list).reshape((len(solvers),len(c_values)))
    print(itr_list)