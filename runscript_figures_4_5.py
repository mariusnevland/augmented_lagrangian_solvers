import numpy as np
import porepy as pp
import copy
from model_setup_two_dim import *
from run_and_report_single import *
from parameters import *
from export_iterations import *
from contact_states_counter import *
from porepy.applications.test_utils.models import add_mixin
from model_setup_common import *

# Runscript for producing figures 5 and 6 in the article.

class SimpleInjectionInit(MoreFocusedFractures,
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
                      MoreFocusedFractures,
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


# Figure 5a
c_value = 1e2
solvers = ["GNM", "IRM", "GNM-RM"]
for solver in solvers:
    params = copy.deepcopy(params_injection_2D)
    params["max_iterations"] = 50
    params["make_fig5a"] = True
    params["injection_overpressure"] = 0.8 * 1e7
    _ = run_and_report_single(Model=SimpleInjection, params=params, c_value=c_value, solver=solver)
plt.legend(["GNM", "IRM", "GNM-RM"], fontsize=14)
plt.xlabel("Iteration", fontsize=14)
plt.ylabel("Residual norm", fontsize=14)
plt.savefig("fig5a.png", dpi=300, bbox_inches="tight")
plt.close()

# Figure 5b
solver = "Delayed_GNM-RM"
params = copy.deepcopy(params_injection_2D)
params["make_fig5b"] = True
params["injection_overpressure"] = 0.8 * 1e7
_ = run_and_report_single(Model=SimpleInjection, params=params, c_value=c_value, solver=solver)
plt.legend(["Return map off", "Return map on"], fontsize=14)
plt.xlabel("Iteration", fontsize=14)
plt.ylabel("Residual norm", fontsize=14)
plt.savefig("fig5b.png", dpi=300, bbox_inches="tight")
plt.close()

# Figure 6
ModelWithContactCounter = add_mixin(ContactStatesCounter, SimpleInjection)
solvers = ["GNM", "GNM-RM", "IRM"]
for solver in solvers:
    params = copy.deepcopy(params_injection_2D)
    params["make_fig6"] = True
    params["max_iterations"] = 30
    params["injection_overpressure"] = 0.8 * 1e7
    _ = run_and_report_single(Model=ModelWithContactCounter, params=params, c_value=c_value, solver=solver)
    plt.xlabel("Iteration", fontsize=14)
    plt.ylabel("Number of cells in contact state", fontsize=14)
    if solver == "Newton":
        plt.legend(["Open", "Stick", "Slip"], fontsize=14, loc=(0.74,0.1))
        plt.title("GNM", fontsize=14)
        plt.savefig("fig6a.png", dpi=300, bbox_inches="tight")
        plt.close()
    elif solver == "NewtonReturnMap":
        plt.legend(["Open", "Stick", "Slip"], fontsize=14)
        plt.title("GNM-RM", fontsize=14)
        plt.savefig("fig6b.png", dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.legend(["Regularized open", "Regularized stick", "Regularized slip"], fontsize=14)
        plt.title("IRM", fontsize=14)
        plt.savefig("fig6c.png", dpi=300, bbox_inches="tight")
        plt.close()