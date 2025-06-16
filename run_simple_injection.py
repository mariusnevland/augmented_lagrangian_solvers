import numpy as np
import porepy as pp
import copy
from heatmap import *
from model_setup_example_1 import *
from run_and_report_single import *
from parameters import *
from cubic_normal_permeability import *
from convergence_metrics import *
from export_iterations import *
from run_and_make_plot import *
from contact_mechanics_mixins import *
from contact_states_counter import *

class SimpleInjectionInit(MoreFocusedFractures,
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

def get_initial_condition():
    params = copy.deepcopy(params_initialization)
    model = SimpleInjectionInit(params)
    pp.run_time_dependent_model(model, params)
    vals = model.equation_system.get_variable_values(iterate_index=0)
    return vals

init_cond = get_initial_condition()
print(np.linalg.norm(init_cond))

class InitialCondition:
    def initial_condition(self) -> None:
        super().initial_condition()
        self.equation_system.set_variable_values(
            init_cond,
            time_step_index=0,
            iterate_index=0,
        )


class SimpleInjection(InitialCondition,
                      MoreFocusedFractures,
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

c_values = [1e-4, 1e-3, 1e-2, 1e-1, 1e0]
solvers = ["NewtonReturnMap"]
injection_pressures = [0.1 * 1e7]

itr_list = []
fig_index = ["b", "c"]
for (pressure, ind) in zip(injection_pressures, fig_index):
    for solver in solvers:
        for c in c_values:
            params = copy.deepcopy(params_plots_2D)
            params["max_iterations"] = 100
            params["injection_overpressure"] = pressure
            itr_solver = run_and_report_single(Model=SimpleInjection, params=params, c_value=c, solver=solver)
            itr_list.append(itr_solver)
            print(itr_solver)
    itr_list = np.array(itr_list).reshape((len(solvers),len(c_values)))
    print(itr_list)
    # c_vals = ["1e-3", "1e-2", "1e-1", "1e0", "1e1", "1e2", "1e3"]
    # solvers_ticks = ["Newton", "NRM", "CRM"]
    # heatmap(data=itr_list, vmin=1, vmax=150, xticks=c_vals, yticks=solvers_ticks,
    #         xlabel="c-parameter", file_name=f"fig4{ind}")
    itr_list = []