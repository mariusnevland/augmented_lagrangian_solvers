import numpy as np
import porepy as pp
import scipy.sparse as sps
from typing import TypeVar, Callable, Optional, cast
from functools import partial
Scalar = pp.ad.Scalar
from porepy.numerics.ad.forward_mode import AdArray
from porepy.models.constitutive_laws import PressureStress
FloatType = TypeVar("FloatType", AdArray, np.ndarray, float)

# Collection of mixins used in both the two- and three-dimensional models.


class ContactMechanicsConstant:

    # Mixin to change the contact mechanics numerical constant

    def contact_mechanics_numerical_constant(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.Scalar:
        val = 1e-1
        return pp.ad.Scalar(val, name="Contact_mechanics_numerical_constant")


class AlternativeTangentialEquation:

    # Use b instead of max(0,b) in the tangential equation. The maximum is redundant, due
    # to the characteristic function.

    # We also change the max-function to use the second argument in case of a tie.
    # Doing so for the tangential function significantly improves the robustness of
    # GNM-RM.

    def tangential_fracture_deformation_equation(
        self,
        subdomains: list[pp.Grid],
    ) -> pp.ad.Operator:
        
        num_cells = sum([sd.num_cells for sd in subdomains])
        nd_vec_to_tangential = self.tangential_component(subdomains)

        tangential_basis = self.basis(subdomains, dim=self.nd - 1)

        scalar_to_tangential = pp.ad.sum_projection_list(tangential_basis)

        t_t: pp.ad.Operator = nd_vec_to_tangential @ self.contact_traction(subdomains)
        u_t: pp.ad.Operator = nd_vec_to_tangential @ self.plastic_displacement_jump(
            subdomains
        )
        u_t_increment: pp.ad.Operator = pp.ad.time_increment(u_t)

        # Vectors needed to express the governing equations
        ones_frac = pp.ad.DenseArray(np.ones(num_cells * (self.nd - 1)))

        f_max = pp.ad.Function(self.maximum, "max_function")
        f_norm = pp.ad.Function(partial(pp.ad.l2_norm, self.nd - 1), "norm_function")

        c_num_as_scalar = self.contact_mechanics_numerical_constant(subdomains)
        tangential_sum = t_t + (scalar_to_tangential @ c_num_as_scalar) * u_t_increment

        norm_tangential_sum = f_norm(tangential_sum)
        norm_tangential_sum.set_name("norm_tangential")

        b_p = self.friction_bound(subdomains)
        b_p.set_name("bp")

        bp_tang = (scalar_to_tangential @ b_p) * tangential_sum

        maxbp_abs = scalar_to_tangential @ f_max(b_p, norm_tangential_sum)
        characteristic = self.contact_mechanics_open_state_characteristic(subdomains)
        equation: pp.ad.Operator = (ones_frac - characteristic) * (
            bp_tang - maxbp_abs * t_t
        ) + characteristic * t_t
        equation.set_name("tangential_fracture_deformation_equation")
        return equation

    def contact_mechanics_open_state_characteristic(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.Operator:
        tangential_basis = self.basis(subdomains, dim=self.nd - 1)
        scalar_to_tangential = pp.ad.sum_projection_list(tangential_basis)
        tol = self.numerical.open_state_tolerance
        f_characteristic = pp.ad.Function(
            partial(pp.ad.functions.characteristic_function, tol),
            "characteristic_function_for_zero_normal_traction",
        )
        b_p = self.friction_bound(subdomains)
        b_p.set_name("bp")

        characteristic: pp.ad.Operator = scalar_to_tangential @ f_characteristic(b_p)
        characteristic.set_name("characteristic_function_of_b_p")
        return characteristic
    
    def maximum(self, var_0: FloatType, var_1: FloatType) -> FloatType:
        """Alternative maximum function that chooses the second argument in case of a tie.

        """
        # If neither var_0 or var_1 are AdArrays, return the numpy maximum function.
        if not isinstance(var_0, AdArray) and not isinstance(var_1, AdArray):
            # FIXME: According to the type hints, this should not be possible.
            return np.maximum(var_0, var_1)

        # Make a fall-back zero Jacobian for constant arguments.
        # EK: It is not clear if this is relevant, or if we filter out these cases with the
        # above parsing of numpy arrays. Keep it for now, but we should revisit once we
        # know clearer how the Ad-machinery should be used.
        zero_jac = 0
        if isinstance(var_0, AdArray):
            zero_jac = sps.csr_matrix(var_0.jac.shape)
        elif isinstance(var_1, AdArray):
            zero_jac = sps.csr_matrix(var_1.jac.shape)

        # Collect values and Jacobians.
        vals = []
        jacs = []
        for var in [var_0, var_1]:
            if isinstance(var, AdArray):
                v = var.val
                j = var.jac
            else:
                v = var
                j = zero_jac
            vals.append(v)
            jacs.append(j)

        # If both are scalar, return same. If one is scalar, broadcast explicitly
        if isinstance(vals[0], (float, int)):
            if isinstance(vals[1], (float, int)):
                # Both var_0 and var_1 are scalars. Treat vals as a numpy array to return
                # the maximum. The Jacobian of a scalar is 0.
                val = np.max(vals)
                return pp.ad.AdArray(val, 0)
            else:
                # var_0 is a scalar, but var_1 is not. Broadcast to shape of var_1.
                vals[0] = np.ones_like(vals[1]) * vals[0]
        if isinstance(vals[1], (float, int)):
            # var_1 is a scalar, but var_0 is not (or else we would have hit the return
            # statement in the above double-if). Broadcast var_1 to shape of var_0.
            vals[1] = np.ones_like(vals[0]) * vals[1]

        # By now, we know that both vals are numpy arrays. Try to convince mypy that this is
        # the case.
        assert isinstance(vals[0], np.ndarray) and isinstance(vals[1], np.ndarray)
        # Maximum of the two arrays
        inds = (vals[1] >= vals[0]).nonzero()[0]

        max_val = vals[0].copy()
        max_val[inds] = vals[1][inds]
        # If both arrays are constant, a 0 matrix has been assigned to jacs.
        # Return here to avoid calling copy on a number (immutable, no copy method) below.
        if isinstance(jacs[0], (float, int)):
            assert np.isclose(jacs[0], 0)
            assert np.isclose(jacs[1], 0)
            return AdArray(max_val, 0)

        # Start from var_0, then change entries corresponding to inds.
        max_jac = jacs[0].copy()

        if isinstance(max_jac, (sps.spmatrix, sps.sparray)):
            # Enforce csr format, unless the matrix is csc, in which case we keep it.
            if not max_jac.getformat() == "csc":
                max_jac = max_jac.tocsr()
            lines = pp.matrix_operations.slice_sparse_matrix(jacs[1].tocsr(), inds)
            pp.matrix_operations.merge_matrices(max_jac, lines, inds, max_jac.getformat())
        else:
            max_jac[inds] = jacs[1][inds]

        return AdArray(max_val, max_jac)
    

class DimensionalContactTraction:

    # Revert the contact traction back to its original, dimensional form, by
    # overwriting the characteristic contact traction.

    def characteristic_contact_traction(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.Operator:
        """Characteristic traction [Pa].

        Parameters:
            subdomains: List of subdomains where the characteristic traction is defined.

        Returns:
            Scalar operator representing the characteristic traction.

        """
        t_char = Scalar(1.0)
        t_char.set_name("characteristic_contact_traction")
        return t_char
    

class LebesgueMetric:  # (BaseMetric):
    """Dimension-consistent Lebesgue metric (blind to physics), but separates dimensions."""

    equation_system: pp.EquationSystem
    """EquationSystem object for the current model. Normally defined in a mixin class
    defining the solution strategy.

    """
    volume_integral: Callable[[pp.ad.Operator, list[pp.Grid], int], pp.ad.Operator]
    """General volume integral operator, defined in `pp.BalanceEquation`."""

    def variable_norm(
        self, values: np.ndarray, variables: Optional[list[pp.ad.Variable]] = None
    ) -> float:
        """Implementation of mixed-dimensional Lebesgue L2 norm of a physical state.

        Parameters:
            values: algebraic respresentation of a mixed-dimensional variable
            variables: list of variables to be considered

        Returns:
            float: measure of values

        """
        # Initialize container for collecting separate L2 norms (squarred).
        integrals_squarred = []

        # Use the equation system to get a view onto mixed-dimensional data structures.
        # Compute the L2 norm of each variable separately, automatically taking into
        # account volume and specific volume
        if variables is None:
            variables = self.equation_system.variables
        for variable in variables:

            # Assume low-order discretization with 1 DOF per active entity
            variable_dim = variable._cells + variable._faces + variable._nodes
            l2_norm = pp.ad.Function(partial(pp.ad.l2_norm, variable_dim), "l2_norm")
            sd = variable.domain
            indices = self.equation_system.dofs_of([variable])
            ad_values = pp.ad.DenseArray(values[indices])
            integral_squarred = np.sum(
                self.volume_integral(l2_norm(ad_values) ** 2, [sd], 1).value(
                    self.equation_system
                )
            )

            # Collect the L2 norm squared.
            integrals_squarred.append(integral_squarred)

        # Squash all results by employing a consistent L2 approach.
        return np.sqrt(np.sum(integrals_squarred))

    def residual_norm(
        self,
        residual: np.ndarray,
    ) -> float:
        """Essentially an Euclidean norm as the residuals already integrate over cells."""
        residual_norm = np.linalg.norm(residual)
        return residual_norm
    

class LebesgueConvergenceMetrics:

    def compute_nonlinear_increment_norm(
        self, nonlinear_increment: np.ndarray
    ) -> float:

        return LebesgueMetric.variable_norm(self, nonlinear_increment)
    
    def compute_residual_norm(self, residual: np.ndarray, reference_residual: np.ndarray) -> float:
        return LebesgueMetric.residual_norm(self, residual)
    

class NormalPermeabilityFromSecondary:
    """Introduce the cubic law for the normal permeability."""

    def normal_permeability(self, interfaces: list[pp.MortarGrid]) -> pp.ad.Operator:
        subdomains = self.interfaces_to_subdomains(interfaces)
        projection = pp.ad.MortarProjections(self.mdg, subdomains, interfaces, dim=1)
        aperture = self.aperture(subdomains)
        permeability = (aperture ** Scalar(2)) / Scalar(12)
        normal_perm = projection.secondary_to_mortar_avg() @ permeability
        return normal_perm


class CustomPressureStress(PressureStress):
    """Remove the reference pressure from the stress tensor, as we do not use a reference stress."""

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