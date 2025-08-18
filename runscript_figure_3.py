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

# Runscript for producing figure 3 in the article.

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

def get_initial_condition():
    params = copy.deepcopy(params_initialization)
    model = SimpleInjectionInit(params)
    pp.run_time_dependent_model(model, params)
    vals = model.equation_system.get_variable_values(iterate_index=0)
    return vals

init_cond = get_initial_condition()

class InitialCondition:
    def initial_condition(self) -> None:
        super().initial_condition()
        self.equation_system.set_variable_values(
            init_cond,
            time_step_index=0,
            iterate_index=0,
        )

class SimpleInjection(InitialCondition,
                      FractureNetwork2D,
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

# Test the solvers across different c-values and injection pressures.

c_values = [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4]
solvers = ["GNM", "GNM-RM", "IRM"]
injection_pressures = [0.001 * 1e7, 0.1 * 1e7, 0.8 * 1e7, 1.0 * 1e7]

itr_list = []
fig_index = ["a", "b", "c", "d"]
for (pressure, ind) in zip(injection_pressures, fig_index):
    for solver in solvers:
        for c in c_values:
            params = copy.deepcopy(params_injection_2D)
            params["injection_overpressure"] = pressure
            itr_solver = run_and_report_single(Model=SimpleInjection, params=params, c_value=c, solver=solver)
            itr_list.append(itr_solver)
    itr_list = np.array(itr_list).reshape((len(solvers),len(c_values)))
    c_vals = ["1e-4", "1e-3", "1e-2", "1e-1", "1e0", "1e1", "1e2", "1e3", "1e4"]
    solvers_ticks = ["GNM", "GNM-RM", "IRM"]
    heatmap(data=itr_list, vmin=1, vmax=100, xticks=c_vals, yticks=solvers_ticks,
            xlabel="c-parameter [GPa/m]", file_name=f"Fig4{ind}", title=f"Injection pressure {20 + pressure/1e6} MPa, " + r"$\psi$=5 degrees")
    itr_list = []


# Vary the dilation angle and run the model again for 30MPa injection pressure

def get_initial_condition_smaller():
    params = copy.deepcopy(params_initialization_smaller_dilation)
    model = SimpleInjectionInit(params)
    pp.run_time_dependent_model(model, params)
    vals = model.equation_system.get_variable_values(iterate_index=0)
    return vals

def get_initial_condition_larger():
    params = copy.deepcopy(params_initialization_larger_dilation)
    model = SimpleInjectionInit(params)
    pp.run_time_dependent_model(model, params)
    vals = model.equation_system.get_variable_values(iterate_index=0)
    return vals

init_cond_smaller = get_initial_condition_smaller()
init_cond_larger = get_initial_condition_larger()


class InitialConditionSmaller:
    def initial_condition(self) -> None:
        super().initial_condition()
        self.equation_system.set_variable_values(
            init_cond_smaller,
            time_step_index=0,
            iterate_index=0,
        )

    
class InitialConditionLarger:
    def initial_condition(self) -> None:
        super().initial_condition()
        self.equation_system.set_variable_values(
            init_cond_larger,
            time_step_index=0,
            iterate_index=0,
        )


class SimpleInjectionSmallerDilation(InitialConditionSmaller,
                                     SimpleInjection):
    pass


class SimpleInjectionLargerDilation(InitialConditionLarger,
                                    SimpleInjection):
    pass

models = [SimpleInjectionSmallerDilation, SimpleInjectionLargerDilation]
params = [params_initialization_smaller_dilation, params_initialization_larger_dilation]
fig_index = ["e", "f"]
dilation_angles = [3, 7]
for (model, param, ind, dil) in zip(models, params, fig_index, dilation_angles):
    for solver in solvers:
        for c in c_values:
            params = copy.deepcopy(param)
            params["injection_overpressure"] = 1.0 * 1e7
            itr_solver = run_and_report_single(Model=model, params=params, c_value=c, solver=solver)
            itr_list.append(itr_solver)
    itr_list = np.array(itr_list).reshape((len(solvers),len(c_values)))
    c_vals = ["1e-4", "1e-3", "1e-2", "1e-1", "1e0", "1e1", "1e2", "1e3", "1e4"]
    solvers_ticks = ["GNM", "GNM-RM", "IRM"]
    heatmap(data=itr_list, vmin=1, vmax=100, xticks=c_vals, yticks=solvers_ticks,
            xlabel="c-parameter [GPa/m]", file_name=f"Fig4{ind}", 
            title="Injection pressure 30 MPa, " + r"$\psi$=" + f"{dil} degrees")
    itr_list = []