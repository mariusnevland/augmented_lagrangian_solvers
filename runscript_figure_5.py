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


# Figure 4a
c_value = 1e2
solvers = ["GNM", "IRM", "GNM-RM"]
for solver in solvers:
    params = copy.deepcopy(params_figure_5)
    params["make_fig4a"] = True
    params["injection_overpressure"] = 0.1 * 1e7
    _ = run_and_report_single(Model=SimpleInjection, params=params, c_value=c_value, solver=solver)
plt.legend(["GNM", "IRM", "GNM-RM"], fontsize=14)
plt.xlabel("Iteration", fontsize=14)
plt.ylabel("Residual norm", fontsize=14)
plt.savefig("Fig4atest.png", dpi=300, bbox_inches="tight")
plt.close()