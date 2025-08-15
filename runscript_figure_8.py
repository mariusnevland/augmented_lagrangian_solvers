import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import porepy as pp
import copy 
from model_setup_two_dim import *
from model_setup_three_dim import *
from plot_utils import *
from parameters import *
from model_setup_common import *
from postprocessing_mixins import *

# Runscript for producing figure 8 in the article.
# Warning: This script will take several weeks to run. Specific cases can be run by modifying c_values, solvers or injection_pressures.


class ThreeDimInjectionInit(EllipticFractureNetwork,
                            LithoStaticTraction3D,
                            HydrostaticPressureBC,
                            HydroStaticPressureInitialization,
                            CustomPressureStress,
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
    # Get the directory of the current script
    script_directory = os.path.dirname(os.path.abspath(__file__))
    # Construct the full path for the file
    file_path = os.path.join(script_directory, "initial_condition")
    np.save(file_path, vals)
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
                        CustomPressureStress,
                        LebesgueConvergenceMetrics,
                        AlternativeTangentialEquation,
                        DimensionalContactTraction,
                        NormalPermeabilityFromSecondary,
                        pp.constitutive_laws.GravityForce,
                        pp.constitutive_laws.CubicLawPermeability,
                        pp.poromechanics.Poromechanics):
    pass

c_values = [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4]
injection_pressures = [0.1 * 1e7, 0.8 * 1e7, 1.0 * 1e7]
solvers = ["GNM", "GNM-RM"]
itr_list = []
for pressure in injection_pressures:
    for solver in solvers:
        for c in c_values:
            params = copy.deepcopy(params_injection_3D)
            params["injection_overpressure"] = pressure
            itr_solver = run_and_report_single(Model=ThreeDimInjection, params=params, c_value=c, solver=solver)
            itr_list.append(itr_solver)
itr_list = np.array(itr_list).reshape(len(solvers) * len(injection_pressures),len(c_values))
xticks = ["1e-4", "1e-3", "1e-2", "1e-1", "1e0", "1e1", "1e2", "1e3", "1e4"]
yticks = ["GNM, inj. pressure 21 MPa", "GNM-RM, inj. pressure 21 MPa",
            "GNM, inj. pressure 28 MPa", "GNM-RM, inj. pressure 28 MPa",
            "GNM, inj. pressure 30 MPa", "GNM-RM, inj. pressure 30 MPa"]
heatmap(data=itr_list, vmin=1, vmax=100, xticks=xticks, yticks=yticks,
        xlabel="c-parameter [GPa/m]", file_name="Fig8")
