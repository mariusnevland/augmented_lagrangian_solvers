import numpy as np
import porepy as pp

# Add a postprocessing step after every nonlinear iteration that functions
# as a return map for the contact forces. If the tangential traction is
# larger in magnitude than the friction bound (whose value depends on
# the current guess of the normal traction), it is projected back to the
# boundary. Similarly, if the normal traction is larger than zero, it is
# projected back to the "boundary" of zero.


class NewtonReturnMap:
    # Note: We perform the return map before the nonlinear iteration (i.e. after the previous iteration)
    # since we want the convergence check to be done on the regular Newton update. It is not done
    # before the very first iteration.
    res_before = []
    res_after = []

    def before_nonlinear_iteration(self) -> None:
        if self.nonlinear_solver_statistics.num_iteration > 0:

            res_before_return = self.equation_system.assemble(evaluate_jacobian=False)
            self.res_before.append(np.linalg.norm(res_before_return))
            print("Before return:")
            print(np.linalg.norm(res_before_return))
            # Evaluate tractions and the friction bound.
            fracture_domains = self.mdg.subdomains(dim=self.nd - 1)

            normal_basis = self.normal_component(fracture_domains)
            tangential_basis = self.tangential_component(fracture_domains)

            t_t = self.tangential_component(fracture_domains) @ self.contact_traction(
                fracture_domains
            )
            t_n = self.normal_component(fracture_domains) @ self.contact_traction(
                fracture_domains
            )
            t_t_eval = t_t.value(self.equation_system)
            t_n_eval = t_n.value(self.equation_system)

            # This will be our updated traction array, based on the results of the
            # return map.
            t_eval_new = np.zeros(self.contact_traction(fracture_domains).size)

            # Find the indices of the global traction array corresponding to the normal and
            # tangential components. This is equivalent to the column position of all nonzero elements
            # of the normal and tangential projection matrices, respectively.

            # print(self.tangential_component(fracture_domains).value(self.equation_system))

            # assert False
            # tang_matrix = self.tangential_component(fracture_domains).value(self.equation_system)
            # norm_matrix = self.normal_component(fracture_domains).value(self.equation_system)
            # print(np.nonzero(tang_matrix))
            # print(np.nonzero(norm_matrix))
            # tang_indices = np.nonzero(tang_matrix)[1]
            # norm_indices = np.nonzero(norm_matrix)[1]

            # If the simulation is three-dimensional, the tangential component
            # is (cell-wise) two-dimensional and is accordingly put into
            # a two-dimensional array.
            if self.nd == 3:
                t_n_new = t_n_eval
                t_t_new = t_t_eval.reshape((int(t_t_eval.size / 2), 2))
            else:
                t_n_new = t_n_eval
                t_t_new = t_t_eval

            # Return map in normal direction.
            for i in range(len(t_n_new)):
                if t_n_new[i] > 0:
                    t_n_new[i] = 0  # Set the normal traction to zero.
            friction_coeff = self.friction_coefficient(fracture_domains).value(
                self.equation_system
            )
            friction_bound = -1.0 * friction_coeff * t_n_new
            # Return map in tangential direction, using the newly
            # obtained normal tractions.
            for i, (bound, t_t) in enumerate(zip(friction_bound, t_t_new)):
                if np.linalg.norm(t_t) > bound:
                    # Tangential traction exceeds friction bound,
                    # project back to the boundary.
                    t_t_new[i] = bound * t_t / np.linalg.norm(t_t)

            # Put normal and tangential tractions in the right positions of the global
            # traction array.

            # Projection matrix (represented as an ArraySlicer) from the full traction
            # to the normal direction.
            normal_restriction = normal_basis._slicer
            # Take the transpose to map back to the full traction space.
            normal_prolongation = normal_restriction.T

            # The tangential projection is essentially a list of Ad projection
            # operators. These are stored as the children (this was the simplest way to
            # get the Ad parsing machinery to collaborate).
            if self.nd == 3:
                tangential_restriction = tangential_basis.children
            else:
                tangential_restriction = [tangential_basis]

            # To get the transporse, we fetch the slicer of each child and take its
            # transpose. This should leave tangential_prolongation as a list of
            # ArraySlicers.
            tangential_prolongation = [e._slicer.T for e in tangential_restriction]

            # Apply the normal projection operator to the normal traction.
            t_eval_new = normal_prolongation @ t_n_new
            # Loop over the tangential projection operators and apply them to the
            # components of the tangential traction.
            for ind, e_i in enumerate(tangential_prolongation):
                # Apply the projection operator to the tangential traction.
                if self.nd == 3:
                    t_eval_new += e_i @ t_t_new[:, ind]
                else:
                    t_eval_new += e_i @ t_t_new

            # Update traction values after having performed the return map.
            self.equation_system.set_variable_values(
                t_eval_new, variables=[self.contact_traction_variable], iterate_index=0
            )

            res_after_return = self.equation_system.assemble(evaluate_jacobian=False)
            self.res_after.append(np.linalg.norm(res_after_return))
            print("After return:")
            print(np.linalg.norm(res_after_return))

        super().before_nonlinear_iteration()