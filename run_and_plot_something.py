import numpy as np
import porepy as pp
import copy
import matplotlib.pyplot as plt
from model_setup_example_1 import *
from run_and_report_single import *
from parameters import *
from cubic_normal_permeability import *
from convergence_metrics import *
from export_iterations import *
from contact_mechanics_mixins import *
from contact_states_counter import *

class SimpleInjectionInit(VerticalHorizontalNetwork,
                          AnisotropicStressBC,
                          ConstantPressureBC,
                          ConstrainedPressureEquaton,
                          LebesgueConvergenceMetrics,
                          AlternativeTangentialEquation,
                          ContactMechanicsConstant,
                          DimensionalContactTraction,
                          # IterationExporting,
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
                      VerticalHorizontalNetwork,
                      PressureConstraintWell,
                      AnisotropicStressBC,
                      ConstantPressureBC,
                      LebesgueConvergenceMetrics,
                      AlternativeTangentialEquation,
                      DimensionalContactTraction,
                      ContactMechanicsConstant2,
                      NewtonReturnMap,
                      ContactStatesCounter,
                      # IterationExporting,
                      NormalPermeabilityFromSecondary,
                      pp.constitutive_laws.CubicLawPermeability,
                      pp.poromechanics.Poromechanics):
    pass


params = copy.deepcopy(params_injection_2D)
model = SimpleInjection(params)
try:
    pp.run_time_dependent_model(model, params)
    print("Num open:")
    print(model.num_open)
    print("Num stick:")
    print(model.num_stick)
    print("Num glide:")
    print(model.num_glide)
    plt.plot(np.arange(0, len(model.num_open)), model.num_open)
    plt.plot(np.arange(0, len(model.num_glide)), model.num_glide)
    plt.plot(np.arange(0, len(model.num_stick)), model.num_stick)
    plt.savefig("contact_states_success", dpi=300, bbox_inches="tight")
except ValueError or RuntimeError:
    print("Num open:")
    print(model.num_open)
    print("Num stick:")
    print(model.num_stick)
    print("Num glide:")
    print(model.num_glide)
    plt.plot(np.arange(0, len(model.num_open)), model.num_open)
    plt.plot(np.arange(0, len(model.num_glide)), model.num_glide)
    plt.plot(np.arange(0, len(model.num_stick)), model.num_stick)
    plt.savefig("contact_states_fail", dpi=300, bbox_inches="tight")