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


class ThreeDimInit(EllipticFractureNetwork,
                   LithoStaticTraction3D,
                   HydrostaticPressureGradient3D,
                   pp.constitutive_laws.GravityForce,
                   NormalPermeabilityFromSecondary,
                   NewtonReturnMapTest,
                   pp.constitutive_laws.CubicLawPermeability,
                   pp.poromechanics.Poromechanics):
    pass


params = params_testing_3D
model = ThreeDimInit(params)
pp.run_time_dependent_model(model, params)