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

# Runscript for producing figure 9 in the article.


class ThreeDimInjectionInit(EllipticFractureNetwork,
                            pp_solvers.IterativeSolverMixin,
                            LithoStaticAnisotopic,
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
                        LithoStaticAnisotopic,
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


c_values = [1e0]
solvers = ["GNM"]
itr_list = [[] for _ in c_values]
itr_time_step_list = [[] for _ in c_values]
itr_linear_list = [[] for _ in c_values]
params_init = copy.deepcopy(params_initialize_pressure_3D)
model_init = ThreeDimInjectionInit(params_init)
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
model_class = add_mixin(InitialCondition, ThreeDimInjection)
for (i, c) in enumerate(c_values):
    for solver in solvers:     
        params = copy.deepcopy(params_injection_3D_hard)
        params["injection_overpressure"] = 0.1 * 1e7
        start = time.time()
        [itr_solver, itr_time_step_list_solver, itr_linear_solver] = run_and_report_single(Model=model_class, params=params, c_value=c, solver=solver)
        end = time.time()
        print(f"c-value: {c}, solver: {solver}, time: {end-start} seconds")
        print(f"Nonlinear iterations: {itr_solver}")
        print(f"Linear iterations: {itr_linear_solver}")
        print(f"Nonlinear iteration list: {itr_time_step_list_solver}")
        itr_list[i].append(itr_solver)
        itr_time_step_list[i].append(itr_time_step_list_solver)
        itr_linear_list[i].append(itr_linear_solver)