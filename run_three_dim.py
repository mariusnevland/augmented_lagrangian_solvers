import numpy as np
import porepy as pp
import copy
from model_setup_example_1 import *
from model_setup_three_dim import *
from run_and_report_single import *
from run_and_make_plot import *
from parameters import *
from cubic_normal_permeability import *
from newton_return_map_test import *
from convergence_metrics import *
from export_iterations import *
from contact_mechanics_mixins import *
from contact_states_counter import *
from heatmap import *


class ThreeDimInjectionInit(EllipticFractureNetwork,
                            LithoStaticTraction3D,
                            HydrostaticPressureBC,
                            HydroStaticPressureInitialization,
                            LebesgueConvergenceMetrics,
                            AlternativeTangentialEquation,
                            ContactMechanicsConstant,
                            DimensionalContactTraction,
                            pp.constitutive_laws.GravityForce,
                            NormalPermeabilityFromSecondary,
                            pp.constitutive_laws.CubicLawPermeability,
                            pp.poromechanics.Poromechanics):
    pass


def get_initial_condition():
    params = copy.deepcopy(params_initialize_pressure_3D)
    model = ThreeDimInjectionInit(params)
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


class ThreeDimInjection(InitialCondition,
                        EllipticFractureNetwork,
                        PressureConstraintWell3D,
                        LithoStaticTraction3D,
                        HydrostaticPressureBC,
                        LebesgueConvergenceMetrics,
                        AlternativeTangentialEquation,
                        DimensionalContactTraction,
                        NormalPermeabilityFromSecondary,
                        pp.constitutive_laws.GravityForce,
                        pp.constitutive_laws.CubicLawPermeability,
                        pp.poromechanics.Poromechanics):
    pass

# c_values = [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]
c_values = [1e-1]
solvers = ["Newton"]
injection_pressures = [0.5*1e7]
# injection_pressures = [0.5 * 1e7, 1e7]
itr_list = []
for pressure in injection_pressures:
    for solver in solvers:
        for c in c_values:
            params = copy.deepcopy(params_testing_3D)
            params["max_iterations"] = 100
            params["injection_overpressure"] = pressure
            params["make_fig9b"] = True
            itr_solver = run_and_report_single(Model=ThreeDimInjection, params=params, c_value=c, solver=solver)
            itr_list.append(itr_solver)
            print(itr_solver)
    itr_list = np.array(itr_list).reshape((len(solvers),len(c_values)))
    xticks = ["1e-3", "1e-2", "1e-1", "1e0", "1e1", "1e2", "1e3"]
    yticks = ["Newton, well pressure 25MPa", "NRM, well pressure 25MPa",
               "Newton, well pressure 30MPa", "NRM, well pressure 30MPa"]
# plt.legend(["Newton", "NRM"], fontsize=14)
# plt.xlabel("Iteration", fontsize=14)
# plt.ylabel("Residual norm", fontsize=14)
# plt.savefig("fig9b.png", dpi=300, bbox_inches="tight")
# heatmap(data=itr_list, vmin=1, vmax=150, xticks=xticks, yticks=yticks,
#         xlabel="Total number of cells", file_name="fig9a")
