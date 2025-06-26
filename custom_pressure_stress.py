import porepy as pp
from porepy.models.constitutive_laws import PressureStress
from typing import cast


class CustomPressureStress(PressureStress):

    def pressure_stress(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        for sd in subdomains:
            # The stress is only defined in matrix subdomains. The stress from fluid
            # pressure in fracture subdomains is handled in :meth:`fracture_stress`.
            if sd.dim != self.nd:
                raise ValueError("Subdomain must be of dimension nd.")

        # No need to accommodate different discretizations for the stress tensor, as we
        # have only one.
        discr = pp.ad.BiotAd(self.stress_keyword, subdomains)
        # The stress is simply found by the scalar_gradient operator, multiplied with
        # the pressure perturbation. The reference pressure is only defined on
        # sd_primary, thus there is no need for a subdomain projection.
        stress: pp.ad.Operator = discr.scalar_gradient(
            self.darcy_keyword
        ) @ self.perturbation_from_reference_new("pressure", subdomains)
        stress.set_name("pressure_stress")
        return stress
    

    def perturbation_from_reference_new(self, name: str, grids: list[pp.Grid]):
        quantity = getattr(self, name)
        # This will throw an error if the attribute is not callable
        quantity_op = cast(pp.ad.Operator, quantity(grids))
        # the reference values are a data class instance storing only numbers
        quantity_ref = cast(pp.number, 0)
        # The casting reflects the expected outcome, and is used to help linters find
        # the set_name method
        quantity_perturbed = quantity_op - pp.ad.Scalar(quantity_ref)
        quantity_perturbed.set_name(f"{name}_perturbation")
        return quantity_perturbed