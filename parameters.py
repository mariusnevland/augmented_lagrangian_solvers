import numpy as np
import porepy as pp
import copy
import pp_solvers

solid_values = {"biot_coefficient": 0.8,
                "density": 2.7e3,
                "friction_coefficient": 0.5,
                "lame_lambda": 1.111e10,
                "shear_modulus": 1.7e10,
                "permeability": 1e-15,
                "normal_permeability": 1e-15,  # Cubic law of tangential perm is inherited
                "porosity": 1e-2,
                "dilation_angle": np.radians(5),
                "residual_aperture": 5e-4,
                "fracture_gap": 0
                }

fluid_values = {"compressibility": 1/2.5 * 1e-9,  # Inverse of bulk modulus
                "density": 1e3,
                "viscosity": 1e-3}

numerical_values = {"characteristic_contact_traction": 1}

reference_values = {"pressure": 2e7}

solid = pp.SolidConstants(**solid_values)
fluid = pp.FluidComponent(**fluid_values)
numerical = pp.NumericalConstants(**numerical_values)
nl_convergence_tol = 1e-8
nl_convergence_tol_res = 1e-8
nl_divergence_tol = 1e5
max_iterations = 30
max_iterations_2D = 801
max_iterations_3D = 801
units = pp.Units(kg=1e9, m=1)
material_constants = {"solid": solid, "fluid": fluid, "numerical": numerical}

linear_solver_2D = {
        "preconditioner_factory": pp_solvers.hm_factory,
        "options": {"ksp_max_it": 300, "ksp_rtol": 1e-11,  
                    "ksp_atol": 1e-11, "ksp_gmres_restart": 300,
                    "pc_hypre_boomeramg_strong_threshold": 0.7, 
                    "mechanics": {
                        "pc_hypre_boomeramg_smooth_type": "ILU", "pc_hypre_boomeramg_ilu_level": 2}
                        },
    }

linear_solver_ilu1 = copy.deepcopy(linear_solver_2D)
linear_solver_ilu1["options"]["mechanics"]["pc_hypre_boomeramg_ilu_level"] = 1

linear_solver_ilu3 = copy.deepcopy(linear_solver_2D)
linear_solver_ilu3["options"]["mechanics"]["pc_hypre_boomeramg_ilu_level"] = 3

linear_solver_3D = {
        "preconditioner_factory": pp_solvers.hm_factory,
        "options": {"ksp_max_it": 300, "ksp_rtol": 1e-11,  
                    "ksp_atol": 1e-11, "ksp_gmres_restart": 300,
                    "pc_hypre_boomeramg_strong_threshold": 0.7, 
                    "mechanics": {
                        "pc_hypre_boomeramg_smooth_type": "ILU", "pc_hypre_boomeramg_ilu_level": 0}
                        },
    }

time_manager_injection = pp.TimeManager(
        schedule=[0, 1 * pp.HOUR, 1 * pp.HOUR + 1 * pp.SECOND, 2 * pp.HOUR,
                  2 * pp.HOUR + 1 * pp.SECOND, 3 * pp.HOUR], dt_init=1 * pp.SECOND, 
        dt_min_max=(0.1 * pp.SECOND, 1 * pp.DAY),
        iter_max=max_iterations,
        iter_optimal_range=(4, 20),
        iter_relax_factors=(0.7, 3.0),
        constant_dt=False, recomp_factor=0.5,
        recomp_max=6, print_info=True
    )


time_manager_injection_3D = pp.TimeManager(
        schedule=[0, 1 * pp.HOUR, 1 * pp.HOUR + 1 * pp.SECOND, 2 * pp.HOUR,
                  2 * pp.HOUR + 1 * pp.SECOND, 3 * pp.HOUR],
        dt_init=1 * pp.SECOND, 
        dt_min_max=(0.1 * pp.SECOND, 1 * pp.DAY),
        iter_max=max_iterations,
        iter_optimal_range=(8, 20),
        iter_relax_factors=(0.7, 3.0),
        constant_dt=False, recomp_factor=0.5,
        recomp_max=6, print_info=False
    )

params_initialization = {
    "max_iterations": max_iterations,
    "material_constants": material_constants,
    "time_manager": pp.TimeManager(
        schedule=[0, 0.1 * pp.DAY], dt_init=0.1 * pp.DAY, constant_dt=True
    ),
    "units": units,
    "nl_convergence_tol": nl_convergence_tol,
    "nl_convergence_tol_res": nl_convergence_tol_res,
    "nl_divergence_tol": nl_divergence_tol,
    "linear_solver": linear_solver_2D,
    "folder_name": "results/initialization",
    "reference_variable_values": pp.ReferenceVariableValues(**reference_values),
}

params_injection_2D = {
    "max_iterations": max_iterations,
    "max_total_iterations": max_iterations_2D,
    "material_constants": material_constants,
    "time_manager": time_manager_injection,
    "units": units,
    "nl_convergence_tol": nl_convergence_tol,
    "nl_convergence_tol_res": nl_convergence_tol_res,
    "nl_divergence_tol": nl_divergence_tol,
    "linear_solver": linear_solver_2D,
    "folder_name": "results/injection_2D",
    "reference_variable_values": pp.ReferenceVariableValues(**reference_values),
}

params_figure_5 = {
    "max_iterations": max_iterations,
    "max_total_iterations": max_iterations_2D,
    "material_constants": material_constants,
    "time_manager": pp.TimeManager(
        schedule=[0, 1 * pp.SECOND], dt_init=1 * pp.SECOND, constant_dt=True
    ),
    "units": units,
    "nl_convergence_tol": nl_convergence_tol,
    "nl_convergence_tol_res": nl_convergence_tol_res,
    "nl_divergence_tol": nl_divergence_tol,
    "linear_solver": linear_solver_2D,
    "folder_name": "results/injection_2D",
    "reference_variable_values": pp.ReferenceVariableValues(**reference_values),
}

params_injection_3D = {
    "max_iterations": max_iterations,
    "max_total_iterations": max_iterations_3D,
    "material_constants": material_constants,
    "time_manager": time_manager_injection_3D,
    "units": units,
    "nl_convergence_tol": nl_convergence_tol,
    "nl_convergence_tol_res": nl_convergence_tol_res,
    "nl_divergence_tol": nl_divergence_tol,
    "linear_solver": linear_solver_3D,
    "folder_name": "results/injection_3D",
    "reference_variable_values": pp.ReferenceVariableValues(**reference_values),
}


params_initialize_pressure_3D = {
    "max_iterations": max_iterations,
    "material_constants": material_constants,
    "time_manager": pp.TimeManager(
        schedule=[0, 50 * pp.DAY], dt_init=50 * pp.DAY, constant_dt=True
    ),
    "units": units,
    "nl_convergence_tol": nl_convergence_tol,
    "nl_convergence_tol_res": nl_convergence_tol_res,
    "linear_solver": linear_solver_3D,
    "folder_name": "results/init_pressure_3D",
    "reference_variable_values": pp.ReferenceVariableValues(**reference_values),
}