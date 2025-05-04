import numpy as np
import porepy as pp
import copy
from model_setup_example_1 import *
from model_setup_three_dim import *
from run_and_report_single import *
from parameters import *
from cubic_normal_permeability import *
from newton_return_map_test import *
from convergence_metrics import *
from export_iterations import *
from contact_mechanics_mixins import *


# class InitializePressure(EllipticFractureNetwork,
#                          HydrostaticPressureGradient3D,
#                          LebesgueConvergenceMetrics,
#                          pp.constitutive_laws.GravityForce,
#                          NormalPermeabilityFromSecondary,
#                          pp.constitutive_laws.CubicLawPermeability,
#                          pp.fluid_mass_balance.SinglePhaseFlow):
#     pass

# params = copy.deepcopy(params_initialize_pressure_3D)
# model = InitializePressure(params)
# pp.run_time_dependent_model(model, params)
# vals = model.equation_system.get_variable_values(iterate_index=0)





# class InitializeMechanics(EllipticFractureNetwork,
#                           LithoStaticTraction3D,
#                           HydrostaticPressureGradient3D,
#                           LebesgueConvergenceMetrics,
#                           AlternativeTangentialEquation,
#                           ContactMechanicsConstant,
#                           DimensionalContactTraction,
#                           pp.constitutive_laws.GravityForce,
#                           NormalPermeabilityFromSecondary,
#                           pp.constitutive_laws.CubicLawPermeability,
#                           pp.poromechanics.Poromechanics):
#     pass

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
print(np.linalg.norm(init_cond))

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
                        IterationExporting,
                        pp.constitutive_laws.GravityForce,
                        pp.constitutive_laws.CubicLawPermeability,
                        pp.poromechanics.Poromechanics):
    pass


c_values = [1e-3]
solvers = ["Newton"]
itr_list = []
for solver in solvers:
    for c in c_values:
        params = copy.deepcopy(params_testing_3D)
        params["max_iterations"] = 100
        itr_solver = run_and_report_single(Model=ThreeDimInjection, params=params, c_value=c, solver=solver)
        itr_list.append(itr_solver)
itr_list = np.array(itr_list).reshape((len(solvers),len(c_values)))
print(itr_list)

