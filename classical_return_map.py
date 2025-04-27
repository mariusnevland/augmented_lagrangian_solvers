import porepy as pp
import numpy as np
from typing import Literal, Optional
from functools import partial
import scipy.sparse as sps
from typing import TypeVar
Scalar = pp.ad.Scalar
from porepy.numerics.ad.forward_mode import AdArray
FloatType = TypeVar("FloatType", AdArray, np.ndarray, float)


# The implicit variant of the Uzawa algorithm, where the contact conditions are not decoupled
# from the rest of the system. 
# NB! To run a model using the implicit Uzawa algorithm, you need to call run_implicit_uzawa_model.

class ClassicalReturnMap:

    total_itr = 0  # Counter for the total number of nonlinear iterations 
                   # (i.e. number of linear systems solved)

    uzawa_itr = 0

    c_n: float  # Regularization parameter

    c_t: float  # Regularization parameter

    t_n_prev: np.ndarray  # Normal traction from previous Uzawa step.

    t_t_prev: np.ndarray  # Tangential traction from previous Uzawa step.

    def contact_mechanics_normal_constant(self, subdomains: list[pp.Grid]) -> pp.ad.Scalar:

        return pp.ad.Scalar(self.c_n, name="contact_mechanics_normal_constant")

    def contact_mechanics_tangential_constant(self, subdomains: list[pp.Grid]) -> pp.ad.Scalar:

        return pp.ad.Scalar(self.c_t, name="contact_mechanics_tangential_constant")
    
    def normal_fracture_deformation_equation(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.Operator:
        """Regularized version of the normal fracture deformation equation."""

        # Variables
        nd_vec_to_normal = self.normal_component(subdomains)
        # The normal component of the contact traction and the displacement jump.
        t_n: pp.ad.Operator = nd_vec_to_normal @ self.contact_traction(subdomains)
        u_n: pp.ad.Operator = nd_vec_to_normal @ self.displacement_jump(subdomains)

        # Maximum function
        num_cells: int = sum([sd.num_cells for sd in subdomains])
        max_function = pp.ad.Function(pp.ad.maximum, "max_function")
        zeros_frac = pp.ad.DenseArray(np.zeros(num_cells), "zeros_frac")
        t_n_prev_ad = pp.ad.DenseArray(self.t_n_prev, "t_n_prev")

        # The complimentarity condition
        equation: pp.ad.Operator = t_n + max_function(
            pp.ad.Scalar(-1.0) * t_n_prev_ad
            - self.contact_mechanics_normal_constant(subdomains)
            * (u_n - self.fracture_gap(subdomains)),
            zeros_frac,
        )
        equation.set_name("normal_fracture_deformation_equation")
        return equation
    
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

        ones_frac = pp.ad.DenseArray(np.ones(num_cells * (self.nd - 1)))
        zeros_frac = pp.ad.DenseArray(np.zeros(num_cells))

        f_max = pp.ad.Function(self.maximum, "max_function")
        f_norm = pp.ad.Function(partial(pp.ad.l2_norm, self.nd - 1), "norm_function")

        c_num_as_scalar = self.contact_mechanics_tangential_constant(subdomains)
        t_t_prev_ad = pp.ad.DenseArray(self.t_t_prev, "t_t_prev")
        tangential_sum = t_t_prev_ad + (scalar_to_tangential @ c_num_as_scalar) * u_t_increment

        norm_tangential_sum = f_norm(tangential_sum)
        norm_tangential_sum.set_name("norm_tangential")

        b_p = f_max(self.friction_bound(subdomains), zeros_frac)
        b_p.set_name("bp")

        bp_tang = (scalar_to_tangential @ b_p) * tangential_sum

        maxbp_abs = scalar_to_tangential @ f_max(b_p, norm_tangential_sum)

        characteristic = self.contact_mechanics_open_state_characteristic(subdomains)

        equation: pp.ad.Operator = (ones_frac - characteristic) * (
            bp_tang - maxbp_abs * t_t
        ) + characteristic * t_t
        equation.set_name("tangential_fracture_deformation_equation")
        return equation
    
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

    def initial_condition(self) -> None:
        super().initial_condition()

        self.c_n_init = float(1e0)
        self.c_t_init = float(1e0)
        self.c_n = self.c_n_init  # Initial numerical constant for the normal equation
        self.c_t = (
            self.c_t_init
        )  # Initial numerical constant for the tangential equation

        # Initial values of normal and tangential traction
        subdomains = self.mdg.subdomains(dim=self.nd - 1)
        t_n = self.normal_component(subdomains) @ self.contact_traction(subdomains)
        t_t = self.tangential_component(subdomains) @ self.contact_traction(subdomains)
        self.t_n_prev = t_n.value(self.equation_system)
        self.t_t_prev = t_t.value(self.equation_system)

    def after_nonlinear_convergence(
        self, iteration_counter: Optional[int] = None
    ) -> None:
        """Some postprocessing is needed before proceeding to the next Uzawa iteration.
        Note also that we do not proceed to the next time step after nonlinear convergence,
        as we also require the outer Uzawa loop to converge.
        """
        # Update numerical constants. Lots of options for how to update.
        self.c_n += 0
        self.c_t += 0

        # Update normal and tangential tractions, to be used as constants in the next
        # Uzawa iteration.
        subdomains = self.mdg.subdomains(dim=self.nd - 1)
        t = self.contact_traction(subdomains)
        t_n = self.normal_component(subdomains) @ self.contact_traction(
            subdomains)
        t_t = self.tangential_component(subdomains) @ self.contact_traction(
            subdomains)
        self.t_prev = t.value(self.equation_system)
        self.t_n_prev = t_n.value(self.equation_system)
        self.t_t_prev = t_t.value(self.equation_system)

        # Update equations with the normal and tangential tractions defined above.
        eqn_lst = list(self.equation_system.equations.keys())
        for eqn in eqn_lst:
            self.equation_system.remove_equation(eqn)
        self.set_equations()

        # Keep track of the total number of nonlinear iterations
        # In other words, the total number of linear systems solved.
        self.total_itr += self.nonlinear_solver_statistics.num_iteration
        self.uzawa_itr += 1

    def after_nonlinear_failure(self) -> None:
        """Method to be called if the inner Newton loop fails to converge."""
        raise ValueError("Inner Newton loop failed to converge.")

    def after_uzawa_convergence(self, iteration_counter: int) -> None:
        """Method to be called after the Uzawa algorithm has converged.

        Note that this method is nearly identical to the after_nonlinear_convergence()
        method in the case of a standard Newton solver. Hence, we have advanced to the
        next time step when this method is called."""

        solution = self.equation_system.get_variable_values(iterate_index=0)
        self.equation_system.shift_time_step_values()
        self.equation_system.set_variable_values(
            values=solution, time_step_index=0, additive=False
        )
        self.convergence_status = True
        self.save_data_time_step()
        # Reset numerical constants, in case an update strategy was used.
        self.c_n = self.c_n_init
        self.c_t = self.c_t_init
        # We do not reset t_n_prev and t_t_prev, as we want the initial
        # guess at the next time step to be the solution at the previous time step.
        print("Uzawa converged")  # To help me keep track during debugging

    def after_uzawa_failure(self) -> None:
        """Method to be called if the outer Uzawa loop fails to converge."""
        self.save_data_time_step()
        raise ValueError("Uzawa iterations did not converge.")