import numpy as np
import porepy as pp
import logging
from newton_return_map import *
from newton_return_map_test import *
from classical_return_map import *
from run_uzawa_model import *

logger = logging.getLogger(__name__)
# Run a simulation with a given nonlinear solver, and report on the number of
# nonlinear iterations.
# If the Newton solver does not converge, it returns 0.
# If the Uzawa solver does not converge (or diverges to infinity) it returns -1.

def run_and_report_single(Model, 
                          params: dict,
                          c_value: float, 
                          solver: str) -> int:
    
    class ContactMechanicsConstant:

            def contact_mechanics_normal_constant(
                    self, subdomains: list[pp.Grid]
            ) -> pp.ad.Scalar:
                return pp.ad.Scalar(c_value, name="Contact_mechanics_normal_constant")

            def contact_mechanics_tangential_constant(
                    self, subdomains: list[pp.Grid]
            ) -> pp.ad.Scalar:
                return pp.ad.Scalar(c_value, name="Contact_mechanics_tangential_constant")

            # Note: We also change the original constant because it is used to measure
            # the residual error for Uzawa.
            def contact_mechanics_numerical_constant(
                    self, subdomains: list[pp.Grid]
            ) -> pp.ad.Scalar:
                return pp.ad.Scalar(c_value, name="Contact_mechanics_numerical_constant")
    
    if solver == "Newton":

        class Simulation(ContactMechanicsConstant,
                         SumTimeSteps,
                         Model):
            pass
    
    elif solver == "ClassicalReturnMap":

        class Simulation(ContactMechanicsConstant,
                         ClassicalReturnMap,
                         Model):
            pass

    elif solver == "NewtonReturnMap":

        class Simulation(ContactMechanicsConstant,
                         SumTimeSteps,
                         NewtonReturnMapTest,
                         # NewtonReturnMap,
                         Model):
            pass
        
    else:
        raise NotImplementedError("Invalid nonlinear solver.")

    model = Simulation(params)
    if solver == "Newton" or solver == "NewtonReturnMap":
        try:
            pp.run_time_dependent_model(model, params)
            itr = model.total_itr
        except ValueError as e:
            logger.warning(f"Value error: {e}")
            itr = 0
        except RuntimeError as e:
            logger.warning(f"Runtime error: {e}")
            itr = 0
    if solver == "ExplicitUzawa" or solver == "ImplicitUzawa":
        try:
            run_time_dependent_uzawa_model(model, params)
            itr = model.total_itr
        except ValueError as e:
            logger.warning(f"Value error: {e}")
            itr = 0
        except RuntimeError as e:
            logger.warning(f"Runtime error: {e}")
            itr = 0
    return itr


# Sum number of nonlinear iterations over several time steps.
class SumTimeSteps:

    total_itr = 0  # Total number of iterations across all time steps.

    def after_nonlinear_convergence(self) -> None:
        super().after_nonlinear_convergence()
        self.total_itr += self.nonlinear_solver_statistics.num_iteration