from functools import partial
import numpy as np
import porepy as pp
Scalar = pp.ad.Scalar


class ContactStatesCounter:

    num_open = []
    num_stick = []
    num_glide = []

    def compute_fracture_states(self, split_output: bool = False) -> list:
        # Compute states of each fracture cell using the displacement increment.
        # Returns a list where:
        # Open=0, Sticking=1, Gliding=2
        subdomains = self.mdg.subdomains(dim=self.nd - 1)
        f_norm = pp.ad.Function(partial(pp.ad.l2_norm, self.nd - 1), "norm_function")
        tangential_basis: list[pp.ad.SparseArray] = self.basis(
            subdomains, dim=self.nd - 1  # type: ignore[call-arg]
        )
        t_n = self.normal_component(subdomains) @ self.contact_traction(subdomains)
        t_t = self.tangential_component(subdomains) @ self.contact_traction(subdomains)
        u_n = self.normal_component(subdomains) @ self.displacement_jump(subdomains)
        u_t = self.tangential_component(subdomains) @ self.displacement_jump(subdomains)
        u_t_increment: pp.ad.Operator = pp.ad.time_increment(u_t)
        c_num_as_scalar = self.contact_mechanics_numerical_constant(subdomains)
        scalar_to_tangential = pp.ad.sum_projection_list(tangential_basis)
        # The fracture states are defined differently for ImplicitReturnMap, as the
        # complementarity functions are regularized in that case.
        if hasattr(self, "t_t_prev"):  # ImplicitReturnMap is used
            t_t_prev_ad = pp.ad.DenseArray(self.t_t_prev)
            tangential_sum = (
                t_t_prev_ad + (scalar_to_tangential @ c_num_as_scalar) * u_t_increment
            )
        else:   # Newton or NewtonReturnMap is used
            tangential_sum = t_t + (scalar_to_tangential @ c_num_as_scalar) * u_t_increment
        norm_tangential_sum = f_norm(tangential_sum)
        b = (
            Scalar(-1.0)
            * self.friction_coefficient(subdomains)
            * (
                t_n
                + self.contact_mechanics_numerical_constant(subdomains)
                * (u_n - self.fracture_gap(subdomains))
            )
        )
        btang = Scalar(-1.0) * self.friction_coefficient(subdomains) * t_n

        norm_tang_sum_eval = norm_tangential_sum.value(self.equation_system)
        b_eval = b.value(self.equation_system)
        states = []
        tol = self.numerical.open_state_tolerance
        for vals_norm, vals_b in zip(norm_tang_sum_eval, b_eval):
            if vals_b - vals_norm > tol and vals_b > tol:
                states.append(1)  # Stick
            elif vals_b - vals_norm <= tol and vals_b > tol:
                states.append(2)  # Glide
            elif vals_b <= tol:
                states.append(0)  # Open
            else:
                print("Should not get here.")

        # Split combined states vector into subdomain-corresponding vectors
        if split_output:
            split_states = []
            num_cells = []
            for sd in subdomains:
                prev_num_cells = int(sum(num_cells))
                split_states.append(
                    np.array(states[prev_num_cells : prev_num_cells + sd.num_cells])
                )
                num_cells.append(sd.num_cells)
            return split_states
        else:
            return states
        
    def before_nonlinear_loop(self) -> None:
        super().before_nonlinear_loop()
        self.num_open.clear()
        self.num_stick.clear()
        self.num_glide.clear()
        
    def after_nonlinear_iteration(self, nonlinear_increment: np.ndarray) -> None:
        super().after_nonlinear_iteration(nonlinear_increment)
        states = self.compute_fracture_states()
        self.num_open.append(states.count(0))
        self.num_stick.append(states.count(1))
        self.num_glide.append(states.count(2))