import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import porepy as pp
import pp_solvers
import time
import copy 
from model_setup_two_dim import *
from model_setup_three_dim import *
from plot_utils import *
from parameters import *
from model_setup_common import *
from postprocessing_mixins import *
from porepy.applications.test_utils.models import add_mixin

# Runscript for producing figure 8 in the article.
# Warning: This script will take several weeks to run. Specific cases can be run by modifying c_values, solvers or injection_pressures.


class ThreeDimInjectionInit(EllipticFractureNetwork,
                            pp_solvers.IterativeSolverMixin,
                            LithoStaticTraction3D,
                            HydrostaticPressureBC,
                            HydroStaticPressureInitialization,
                            CustomPressureStress,
                            LebesgueConvergenceMetrics,
                            AlternativeTangentialEquation,
                            ContactMechanicsConstant,
                            DimensionalContactTraction,
                            pp.constitutive_laws.GravityForce,
                            NormalPermeabilityFromSecondary,
                            pp.constitutive_laws.CubicLawPermeability,
                            pp.poromechanics.Poromechanics):
    pass


class ThreeDimInjection(EllipticFractureNetwork,
                        pp_solvers.IterativeSolverMixin,
                        PressureConstraintWell3D,
                        CustomExporter,
                        ExportInjectionCell,
                        LithoStaticTraction3D,
                        HydrostaticPressureBC,
                        CustomPressureStress,
                        LebesgueConvergenceMetrics,
                        AlternativeTangentialEquation,
                        DimensionalContactTraction,
                        NormalPermeabilityFromSecondary,
                        pp.constitutive_laws.GravityForce,
                        pp.constitutive_laws.CubicLawPermeability,
                        pp.poromechanics.Poromechanics):
    pass

c_values = [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]
solvers = ["GNM", "GNM-RM", "IRM"]
model_index = ["A", "B", "C"]
nonlinearities = ["Constant fracture volume and permeability", "Constant fracture permeability", "Full model"]
for (ind, nonlin) in zip(model_index, nonlinearities):
    itr_list = [[] for _ in c_values]
    itr_time_step_list = [[] for _ in c_values]
    itr_linear_list = [[] for _ in c_values]
    params_init = copy.deepcopy(params_initialize_pressure_3D)
    if nonlin == "Constant fracture volume and permeability":
        model_class_init = add_mixin(ConstantAperture, ThreeDimInjectionInit)
    elif nonlin == "Constant fracture permeability":
        model_class_init = add_mixin(ConstantCubicLawPermeability, ThreeDimInjectionInit)
    elif nonlin == "Full model":
        model_class_init = ThreeDimInjectionInit
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
    if nonlin == "Constant fracture volume and permeability":
        model_class = add_mixin(InitialCondition, add_mixin(ConstantAperture, ThreeDimInjection))
    elif nonlin == "Constant fracture permeability":
        model_class = add_mixin(InitialCondition, add_mixin(ConstantCubicLawPermeability, ThreeDimInjection))
    elif nonlin == "Full model":
        model_class = add_mixin(InitialCondition, ThreeDimInjection) 
    for (i, c) in enumerate(c_values):
        for solver in solvers:     
            params = copy.deepcopy(params_injection_3D)
            params["injection_overpressure"] = 0.1 * 1e7
            params["irm_update_strategy"] = True
            [itr_solver, itr_time_step_list_solver, itr_linear_solver] = run_and_report_single(Model=model_class, params=params, c_value=c, solver=solver)
            itr_list[i].append(itr_solver)
            itr_time_step_list[i].append(itr_time_step_list_solver)
            itr_linear_list[i].append(itr_linear_solver)
    bar_chart(itr_time_step_list, itr_linear_list, ymin=0, ymax=800, 
              num_xticks=len(c_values), labels=[f"{x:.0e}" for x in c_values], 
              file_name=f"bar_chart_model_{ind}_3D", title=f"Model {ind}: {nonlin}")