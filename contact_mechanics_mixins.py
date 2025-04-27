import numpy as np
import porepy as pp
import scipy.sparse as sps
from typing import TypeVar
from functools import partial
Scalar = pp.ad.Scalar
from porepy.numerics.ad.forward_mode import AdArray
FloatType = TypeVar("FloatType", AdArray, np.ndarray, float)

class ContactMechanicsConstant:

    # Mixin to change the contact mechanics numerical constant

    def contact_mechanics_numerical_constant(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.Scalar:
        val = 1e-1
        return pp.ad.Scalar(val, name="Contact_mechanics_numerical_constant")
    
    
class ContactMechanicsConstant2:

    # In case you need two different constants.

    def contact_mechanics_numerical_constant(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.Scalar:
        val = 1e-2
        return pp.ad.Scalar(val, name="Contact_mechanics_numerical_constant")

class AlternativeTangentialEquation:

    # Use b instead of max(0,b) in the tangential equation. The maximum is redundant, due
    # to the characteristic function.

    # We also change the max-function to use the second argument in case of a tie.
    # Doing so for the tangential function significantly improves the robustness of
    # the Newton solver with a return map.

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
        """Ad maximum function represented as an AdArray.

        The maximum function is defined as the element-wise maximum of two arrays.
        At equality, the Jacobian is taken from the first argument. The order of the
        arguments may be important, since it determines which Jacobian is used in
        the case of equality.

        The arguments can be either AdArrays or ndarrays, this duality is needed to allow
        for parsing of operators that can be taken at the current iteration (in which case
        it will parse as an AdArray) or at the previous iteration or time step (in which
        case it will parse as a numpy array).


        Parameters:
            var_0: First argument to the maximum function.
            var_1: Second argument.

            If one of the input arguments is scalar, broadcasting will be used.


        Returns:
            The maximum of the two arguments, taken element-wise in the arrays. The return
            type is AdArray if at least one of the arguments is an AdArray, otherwise it
            is an ndarray. If an AdArray is returned, the Jacobian is computed according to
            the maximum values of the AdArrays (so if element ``i`` of the maximum is
            picked from ``var_0``, row ``i`` of the Jacobian is also picked from the
            Jacobian of ``var_0``). If ``var_0`` is a ndarray, its Jacobian is set to zero.

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