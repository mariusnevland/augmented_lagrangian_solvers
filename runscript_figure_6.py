import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import porepy as pp
import copy
from model_setup_two_dim import *
from plot_utils import *
from parameters import *
from postprocessing_mixins import *
from model_setup_common import *

# Runscript for producing figure 6 in the article.
# Testing the scalability of GNM and GNM-RM with respect to the grid size.

grid_sizes = [175 * 0.07, 125 * 0.07, 90 * 0.07, 70 * 0.07]
injection_pressures = [0.1 * 1e7, 1.0 * 1e7]
solvers = ["GNM", "GNM-RM"]
itr_list = []

for size in grid_sizes:

    class Grid:

        def meshing_arguments(self) -> dict:
            mesh_args = {"cell_size": size / self.units.m}
            return mesh_args
        
    class SimpleInjectionInit(Grid,
                          FractureNetwork2D,
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

    params = copy.deepcopy(params_initialization)
    model = SimpleInjectionInit(params)
    pp.run_time_dependent_model(model, params)
    vals = model.equation_system.get_variable_values(iterate_index=0)

    class InitialCondition:
        def initial_condition(self) -> None:
            super().initial_condition()
            self.equation_system.set_variable_values(
                vals,
                time_step_index=0,
                iterate_index=0,
            )

    class SimpleInjection(InitialCondition,
                      Grid,
                      FractureNetwork2D,
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

    for pressure in injection_pressures:
        for solver in solvers:
            params = copy.deepcopy(params_grid_refinement_2D)
            params["injection_overpressure"] = pressure
            itr_solver = run_and_report_single(Model=SimpleInjection, params=params, c_value=1e-2, solver=solver)
            itr_list.append(itr_solver)
            print(itr_solver)

    # Run also the 30 MPa injection pressure case with a dilation angle of 3 degrees.

    params = copy.deepcopy(params_initialization_smaller_dilation)
    model = SimpleInjectionInit(params)
    pp.run_time_dependent_model(model, params)
    vals_new_dilation = model.equation_system.get_variable_values(iterate_index=0)

    class InitialConditionSmallerDilation:
        def initial_condition(self) -> None:
            super().initial_condition()
            self.equation_system.set_variable_values(
                vals_new_dilation,
                time_step_index=0,
                iterate_index=0,
            )


    class SimpleInjectionSmallerDilation(InitialConditionSmallerDilation,
                      Grid,
                      FractureNetwork2D,
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

    for solver in solvers:
        params = copy.deepcopy(params_grid_refinement_smaller_dilation)
        params["injection_overpressure"] = 1.0 * 1e7
        itr_solver = run_and_report_single(Model=SimpleInjectionSmallerDilation, params=params, c_value=1e-1, solver=solver)
        itr_list.append(itr_solver)
        print(itr_solver)
itr_list = np.array(itr_list).reshape((4, 6)).T
xticks = ["33830", "64068", "121501", "199282"]
yticks = [r"GNM, inj. pressure 21MPa, c=1e-2, $\psi=5^{\circ}$", r"GNM-RM, inj. pressure 21MPa, c=1e-2, $\psi=5^{\circ}$",
               r"GNM, inj. pressure 30MPa, c=1e-2, $\psi=5^{\circ}$", r"GNM-RM, inj. pressure 30MPa, c=1e-2, $\psi=5^{\circ}$",
               r"GNM, inj. pressure 30MPa, c=1e-1, $\psi=3^{\circ}$", r"GNM-RM, inj. pressure 30MPa, c=1e-1, $\psi=3^{\circ}$"]
heatmap(data=itr_list, vmin=1, vmax=200, xticks=xticks, yticks=yticks,
            xlabel="Total number of cells", file_name="Fig6")
