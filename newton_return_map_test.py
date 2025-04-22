"""Newton return map for contact tractions as preprocessing step.

Add a preprocessing step before every nonlinear iteration that functions
as a return map for the contact forces. If the tangential traction is
larger in magnitude than the friction bound (whose value depends on
the current guess of the normal traction), it is projected back to the
boundary. Similarly, if the normal traction is larger than zero, it is
projected back to the "boundary" of zero.

"""

import numpy as np


class NewtonReturnMapTest:
    """Preprocessing step for the Newton solver to perform a return map.

    We perform the return map before the nonlinear iteration, since we want
    the convergence check to be done on the regular Newton update. It is not
    done before the very first iteration.

    """

    def before_nonlinear_iteration(self) -> None:
        if self.nonlinear_solver_statistics.num_iteration > 0:

            res_before_return = self.equation_system.assemble(evaluate_jacobian=False)
            print("Before return:")
            print(np.linalg.norm(res_before_return))
            # Evaluate tractions and the friction bound.
            fracture_domains = self.mdg.subdomains(dim=self.nd - 1)

            # Evaluate normal traction.
            t_n = self.normal_component(fracture_domains) @ self.contact_traction(
                fracture_domains
            )
            t_n_eval = t_n.value(self.equation_system)

            # Return map in normal direction.
            t_n_new = np.clip(t_n_eval, None, 0.0)

            # Put normal in the right positions of the global traction array.
            # Projection matrix (represented as an ArraySlicer) from the full
            # traction to the normal direction. The prolongation operator is
            # the transpose
            normal_basis = self.normal_component(fracture_domains)
            normal_restriction = normal_basis._slicer
            normal_prolongation = normal_restriction.T
            t_eval_new = normal_prolongation @ t_n_new

            # Evaluate tangential traction and split into its components.
            t_t = self.tangential_component(fracture_domains) @ self.contact_traction(
                fracture_domains
            )
            t_t_eval = t_t.value(self.equation_system)
            t_t_new = t_t_eval.reshape((-1, self.nd - 1))

            # Return map in tangential direction, using the newly obtained normal
            # tractions.
            friction_coeff = self.friction_coefficient(fracture_domains).value(
                self.equation_system
            )
            friction_bound = -friction_coeff * t_n_new

            # Use a trust radius to limit the tangential traction - by default, use
            # the physical bounds, but could be set to a larger value to allow for
            # relaxation.
            trust_radius = 1.0 * friction_bound

            # To avoid numerical issues, we set the tangential traction explicitly
            # to zero if the normal traction is zero. This follows the same principle
            # as in the definition of the tangential fracture deformation equation.
            for i, (bound, t_t) in enumerate(zip(trust_radius, t_t_new)):
                if np.linalg.norm(t_t) > bound:
                    # Tangential traction exceeds friction bound,
                    # project back to the boundary.
                    t_t_new[i] = bound * t_t / np.linalg.norm(t_t)

            # Put tangential tractions in the right positions of the global traction
            # array. The tangential projection is essentially a list of Ad projection
            # operators. These are stored as the children (this was the simplest way
            # to get the Ad parsing machinery to collaborate).
            tangential_basis = self.tangential_component(fracture_domains)
            if self.nd == 3:
                tangential_restriction = tangential_basis.children
            else:
                tangential_restriction = [tangential_basis]

            # To get the transpose, we fetch the slicer of each child and take its
            # transpose. This should leave tangential_prolongation as a list of
            # ArraySlicers. Again, a apply the projection operator to the tangential
            # traction.
            tangential_prolongation = [e._slicer.T for e in tangential_restriction]
            for ind, e_i in enumerate(tangential_prolongation):
                t_eval_new += e_i @ t_t_new[:, ind]

            # Update traction values after having performed the return map.
            self.equation_system.set_variable_values(
                t_eval_new, variables=[self.contact_traction_variable], iterate_index=0
            )

            res_after_return = self.equation_system.assemble(evaluate_jacobian=False)
            print("After return:")
            print(np.linalg.norm(res_after_return))

        super().before_nonlinear_iteration()