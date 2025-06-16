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
                      ContactMechanicsConstant2,
                      ContactStatesCounter,
                      NormalPermeabilityFromSecondary,
                      pp.constitutive_laws.CubicLawPermeability,
                      pp.poromechanics.Poromechanics):
    pass


class SimpleInjectionNRM(InitialCondition,
                      MoreFocusedFractures,
                      PressureConstraintWell,
                      AnisotropicStressBC,
                      ConstantPressureBC,
                      LebesgueConvergenceMetrics,
                      AlternativeTangentialEquation,
                      DimensionalContactTraction,
                      ContactMechanicsConstant2,
                      CycleCheck,
                      DelayedNewtonReturnMap,
                      ContactStatesCounter,
                      NormalPermeabilityFromSecondary,
                      pp.constitutive_laws.CubicLawPermeability,
                      pp.poromechanics.Poromechanics):
    pass

models = [SimpleInjectionNRM]
for model in models:
    params = copy.deepcopy(params_plots_2D)
    params["max_iterations"] = 50
    model = model(params)
    try:
        pp.run_time_dependent_model(model, params)
        print("Num open:")
        print(model.num_open)
        print("Num stick:")
        print(model.num_stick)
        print("Num glide:")
        print(model.num_glide)
        return_map_switch = 20
        itr = np.arange(0, len(model.num_open))
        plt.plot(itr[:return_map_switch+1], model.num_open[:return_map_switch+1], color="orange")
        plt.plot(itr[:return_map_switch+1], model.num_stick[:return_map_switch+1], color="red")
        plt.plot(itr[:return_map_switch+1], model.num_glide[:return_map_switch+1], color="blue")
        plt.plot(itr[return_map_switch:], model.num_open[return_map_switch:], linestyle="--", color="orange")
        plt.plot(itr[return_map_switch:], model.num_stick[return_map_switch:], linestyle="--", color="red")
        plt.plot(itr[return_map_switch:], model.num_glide[return_map_switch:], linestyle="--", color="blue")
    except ValueError or RuntimeError:
        print("Num open:")
        print(model.num_open)
        print("Num stick:")
        print(model.num_stick)
        print("Num glide:")
        print(model.num_glide)
        plt.plot(model.num_open, color="orange")
        plt.plot(model.num_stick, color="red")
        plt.plot(model.num_glide, color="blue")
plt.legend(["Open", "Stick", "Slip"])
plt.xlabel("Iteration")
plt.ylabel("Number of cells in contact state")
plt.savefig("contact_states_success", dpi=300, bbox_inches="tight")