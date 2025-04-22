import numpy as np
import porepy as pp
from functools import partial
Scalar = pp.ad.Scalar

class ContactMechanicsConstant:

    # Mixin to change the contact mechanics numerical constant

    def contact_mechanics_numerical_constant(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.Scalar:
        val = 1e-3
        return pp.ad.Scalar(val, name="Contact_mechanics_numerical_constant")

class AlternativeTangentialEquation:

    # Use b instead of max(0,b) in the tangential equation. The maximum is redundant, due
    # to the characteristic function.

    def tangential_fracture_deformation_equation(
        self,
        subdomains: list[pp.Grid],
    ) -> pp.ad.Operator:
        """Contact mechanics equation for the tangential constraints.

        The equation is dimensionless, as we use nondimensionalized contact traction.
        The function reads
        .. math::
            C_t = max(b_p, ||T_t+c_t u_t||) T_t - max(0, b_p) (T_t+c_t u_t)

        with `u` being displacement jump increments, `t` denoting tangential component
        and `b_p` the friction bound.

        For `b_p = 0`, the equation `C_t = 0` does not in itself imply `T_t = 0`, which
        is what the contact conditions require. The case is handled through the use of a
        characteristic function.

        Parameters:
            subdomains: List of fracture subdomains.

        Returns:
            complementary_eq: Contact mechanics equation for the tangential constraints.

        """
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

        f_max = pp.ad.Function(pp.ad.maximum, "max_function")
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