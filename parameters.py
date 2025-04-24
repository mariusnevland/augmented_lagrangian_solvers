import numpy as np
import porepy as pp

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

solid = pp.SolidConstants(**solid_values)
fluid = pp.FluidComponent(**fluid_values)
numerical = pp.NumericalConstants(**numerical_values)
nl_convergence_tol = 1e-8
nl_convergence_tol_res = 1e-8
units = pp.Units(kg=1e9, m=1)
material_constants = {"solid": solid, "fluid": fluid, "numerical": numerical}

params_initialization = {
    "max_iterations": 100,
    "material_constants": material_constants,
    "time_manager": pp.TimeManager(
        schedule=[0, 10 * pp.DAY], dt_init=10 * pp.DAY, constant_dt=True
    ),
    "units": units,
    "nl_convergence_tol": nl_convergence_tol,
    "nl_convergence_tol_res": nl_convergence_tol_res,
    "linear_solver": "scipy_sparse",
    "folder_name": "results/initialization",
}

params_injection_2D = {
    "max_iterations": 100,
    "material_constants": material_constants,
    "time_manager": pp.TimeManager(
        schedule=[0, 0.1 * pp.DAY], dt_init=0.1 * pp.DAY, constant_dt=True
    ),
    "units": units,
    "nl_convergence_tol": nl_convergence_tol,
    "nl_convergence_tol_res": nl_convergence_tol_res,
    "linear_solver": "scipy_sparse",
    "folder_name": "results/injection_2D",
}


params_testing_3D = {
    "max_iterations": 100,
    "material_constants": material_constants,
    "time_manager": pp.TimeManager(
        schedule=[0, 150 * pp.DAY], dt_init=50 * pp.DAY, constant_dt=True
    ),
    "units": units,
    "nl_convergence_tol": nl_convergence_tol,
    "nl_convergence_tol_res": nl_convergence_tol_res,
    "linear_solver": "scipy_sparse",
    "folder_name": "results/testing_3D",
}