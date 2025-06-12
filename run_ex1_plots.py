import numpy as np
import porepy as pp
import copy
from model_setup_example_1 import *
from run_and_report_single import *
from parameters import *
from cubic_normal_permeability import *
from convergence_metrics import *
from export_iterations import *
from run_and_make_plot import *
from contact_mechanics_mixins import *
from contact_states_counter import *
from porepy.applications.test_utils.models import add_mixin


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
                      NormalPermeabilityFromSecondary,
                      pp.constitutive_laws.CubicLawPermeability,
                      pp.poromechanics.Poromechanics):
    pass


# Figure 5a
c_value = 1e2
solvers = ["Newton", "NewtonReturnMap", "ClassicalReturnMap"]
for solver in solvers:
    params = copy.deepcopy(params_plots_2D)
    params["max_iterations"] = 50
    params["make_fig5a"] = True
    params["injection_overpressure"] = 0.1 * 1e7
    _ = run_and_report_single(Model=SimpleInjection, params=params, c_value=c_value, solver=solver)
plt.legend(["Newton", "NRM", "CRM"], fontsize=14)
plt.xlabel("Iteration", fontsize=14)
plt.ylabel("Residual norm", fontsize=14)
plt.savefig("fig5a.png", dpi=300, bbox_inches="tight")
plt.close()

# Figure 5b
solver = "DelayedNewtonReturnMap"
params = copy.deepcopy(params_plots_2D)
params["make_fig5b"] = True
params["injection_overpressure"] = 0.1 * 1e7
_ = run_and_report_single(Model=SimpleInjection, params=params, c_value=c_value, solver=solver)
plt.legend(["Return map on", "Return map off"], fontsize=14)
plt.xlabel("Iteration", fontsize=14)
plt.ylabel("Residual norm", fontsize=14)
plt.savefig("fig5b.png", dpi=300, bbox_inches="tight")
plt.close()

# Figure 6
ModelWithContactCounter = add_mixin(ContactStatesCounter, SimpleInjection)
solvers = ["Newton", "NewtonReturnMap", "ClassicalReturnMap"]
for solver in solvers:
    params = copy.deepcopy(params_plots_2D)
    params["make_fig6"] = True
    params["max_iterations"] = 30
    params["injection_overpressure"] = 0.1 * 1e7
    _ = run_and_report_single(Model=ModelWithContactCounter, params=params, c_value=c_value, solver=solver)
    plt.xlabel("Iteration", fontsize=14)
    plt.ylabel("Number of cells in contact state", fontsize=14)
    if solver == "Newton":
        plt.legend(["Open", "Stick", "Slip"], fontsize=14)
        plt.title("Newton", fontsize=14)
        plt.savefig("fig6a.png", dpi=300, bbox_inches="tight")
        plt.close()
    elif solver == "NewtonReturnMap":
        plt.legend(["Open", "Stick", "Slip"], fontsize=14)
        plt.title("NRM", fontsize=14)
        plt.savefig("fig6b.png", dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.legend(["Regularized open", "Regularized stick", "Regularized slip"], fontsize=14)
        plt.title("CRM", fontsize=14)
        plt.savefig("fig6c.png", dpi=300, bbox_inches="tight")
        plt.close()