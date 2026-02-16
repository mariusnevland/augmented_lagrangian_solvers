import numpy as np
import porepy as pp
import copy
import pp_solvers
from model_setup_two_dim import *
from plot_utils import *
from parameters import *
from postprocessing_mixins import *
from porepy.applications.test_utils.models import add_mixin
from model_setup_common import *

# Runscript for producing figures 4 and 5 in the article.

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
                      PressureConstraintWellGrid,
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
c_value = 1e3
solvers = ["GNM", "IRM", "GNM-RM"]
params_init = copy.deepcopy(params_initialization)
model_class_init = add_mixin(ConstantAperture, SimpleInjectionInit)
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
model_class = add_mixin(InitialCondition, add_mixin(ConstantAperture, SimpleInjection))
for solver in solvers:
    params = copy.deepcopy(params_figure_5)
    params["make_fig5"] = True
    params["injection_overpressure"] = 0.1 * 1e7
    if solver == "IRM":
        params["linear_solver"] = linear_solver_ilu0
    _ = run_and_report_single(Model=model_class, params=params, c_value=c_value, solver=solver)
plt.legend(solvers, fontsize=14)
plt.xlabel("Nonlinear iteration", fontsize=14)
plt.ylabel("Residual norm", fontsize=14)
plt.title("Model A, " + r"$c=10^{3}$ " + "GPa/m", fontsize=14)
plt.savefig("Fig5a.png", dpi=300, bbox_inches="tight")
plt.close()


# Figure 5b
c_value = 1e-3
params_init = copy.deepcopy(params_initialization)
model_init = SimpleInjectionInit(params_init)
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
model_class = add_mixin(InitialCondition, SimpleInjection)
for solver in solvers:
    params = copy.deepcopy(params_figure_5)
    params["make_fig5"] = True
    params["injection_overpressure"] = 0.1 * 1e7
    _ = run_and_report_single(Model=model_class, params=params, c_value=c_value, solver=solver)
plt.legend(solvers, fontsize=14)
plt.xlabel("Nonlinear iteration", fontsize=14)
plt.ylabel("Residual norm", fontsize=14)
plt.title("Model C, " + r"$c=10^{-3}$ " + "GPa/m", fontsize=14)
plt.savefig("Fig5b.png", dpi=300, bbox_inches="tight")
plt.close()