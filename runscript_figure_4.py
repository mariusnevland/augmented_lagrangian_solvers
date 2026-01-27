import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import porepy as pp
import pp_solvers
import copy
import time
from porepy.viz.data_saving_model_mixin import FractureDeformationExporting
from model_setup_two_dim import *
from plot_utils import *
from parameters import *
from linesearch import *
from model_setup_common import *
from postprocessing_mixins import *
from porepy.applications.test_utils.models import add_mixin

# Runscript for producing figure 4 in the article.

class SimpleInjectionInit(FractureNetwork2D,
                          pp_solvers.IterativeSolverMixin,
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
                      pp_solvers.IterativeSolverMixin,
                      CustomExporter,
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


c_values = [1e-2]
solvers = ["GNM"]
itr_list = [[] for _ in c_values]
itr_time_step_list = [[] for _ in c_values]
itr_linear_list = [[] for _ in c_values]
nonlinearities = ["Full model"]
for nonlin in nonlinearities:
    params_init = copy.deepcopy(params_initialization)
    if nonlin == "No aperture":
        model_class_init = add_mixin(ConstantAperture, SimpleInjectionInit)
    elif nonlin == "No cubic law":
        model_class_init = add_mixin(ConstantCubicLawPermeability, SimpleInjectionInit)
    elif nonlin == "Full model":
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
    if nonlin == "No aperture":
        model_class = add_mixin(InitialCondition, add_mixin(ConstantAperture, SimpleInjection))
    elif nonlin == "No cubic law":
        model_class = add_mixin(InitialCondition, add_mixin(ConstantCubicLawPermeability, SimpleInjection))
    elif nonlin == "Full model":
        model_class = add_mixin(InitialCondition, SimpleInjection) 
    for (i, c) in enumerate(c_values):
        for solver in solvers:     
            params = copy.deepcopy(params_injection_2D)
            # params["folder_name"] = "results/injection_2D_states_aperture"
            params["injection_overpressure"] = 0.1 * 1e7
            params["irm_update_strategy"] = True
            start = time.time()
            [itr_solver, itr_time_step_list_solver, itr_linear_solver] = run_and_report_single(Model=model_class, params=params, c_value=c, solver=solver)
            end = time.time()
            print("Time:")
            print(end-start)
            itr_list[i].append(itr_solver)
            itr_time_step_list[i].append(itr_time_step_list_solver)
            itr_linear_list[i].append(itr_linear_solver)
        print(f"c-value: {c}")
        print(itr_time_step_list[i])
        print(itr_list[i])
        print(itr_linear_list[i])
    # bar_chart(itr_time_step_list, itr_linear_list, ymin=0, ymax=500, 
    #           num_xticks=len(c_values), labels = [f"{x:.0e}" for x in c_values], 
    #           file_name="bar_test_no_cubic", title=f"Nonlinearity: {nonlin}")