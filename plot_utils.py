import numpy as np
import porepy as pp
from newton_return_map import *
from implicit_return_map import *
import pandas as pd
import matplotlib.pyplot as plt

"""Various utility functions for plotting results."""

def bar_chart(itr_time_step_list, lin_list, ymin, ymax, num_xticks, labels, file_name, 
              title=None, grid_study=False):
    width = 0.1
    diverged = False
    indices = ['A', 'B', 'C']
    positions = [i * 0.5 for i in range(num_xticks)]
    positions_grid = [0.5*(i + 2*offset) for i in range(num_xticks) for offset in [-width, 0, width]]
    positions2 = [-1, 0, 1]
    colors = [["#0059FF", "#03b7ff", "#00fff7"], 
              ["#ff0000", "#FF00C3", "#c800ff"],
              ["#A26105", "#FABA09FF", "#FFEE06FF"]]
    _, ax1 = plt.subplots()
    ax1.set_ylim(ymin, ymax)
    _, ymax = ax1.get_ylim()
    ax2 = ax1.twinx()
    for (i, pos) in enumerate(positions):
        df = pd.DataFrame(itr_time_step_list[i], index=indices)
        for j, (ind, lin, foo) in enumerate(zip(indices, lin_list[i], positions2)):
            bottom = 0
            color_counter = 0
            for value in df.loc[ind].dropna():
                if value == -1:  # Onset of phase 2
                    color_counter = 1
                    ax1.bar(pos + foo * width, 1, width, align='center', bottom=bottom, linewidth=0.5, color='black')
                    bottom += 1
                    continue
                elif value == -2:  # Onset of phase 3
                    color_counter = 2
                    ax1.bar(pos + foo * width, 1, width, align='center', bottom=bottom, linewidth=0.5, color='black')
                    bottom += 1
                    continue
                elif value == -500:  # Diverged to infinity
                    diverged = True
                    continue
                color = colors[j][color_counter]
                if value==31 or diverged==True:
                    bar = ax1.bar(pos + foo * width, value, width, align='center', bottom=bottom, hatch='/', linewidth=0.5, color=color, hatch_linewidth=10)
                    diverged = False
                else:
                    bar = ax1.bar(pos + foo * width, value, width, align='center', bottom=bottom, linewidth=0.5, color=color)
                bottom += value
            if bottom > ymax:
                rect = bar[0]
                xmid = rect.get_x() + rect.get_width() / 2
                ax1.plot(xmid, ymax * 1.01, marker=(3,0,0), markersize=12, color='red', clip_on=False)
            else:
                vals = df.loc[ind].dropna().values
                # First case below is when the solver used the max amount of allowed iterations.
                # Second case below is when the solver diverged to infinity.
                if not vals[-1] == 31 and not vals[-2] == -500:
                    ax2.plot(pos + foo * width, lin, marker='o', color='black')
    if grid_study==True:
        plt.xticks(positions_grid, labels)
    else:
        plt.xticks(positions, labels)
    ax1.set_xlabel("c-value [GPa/m]")
    ax1.set_ylabel("Nonlinear iterations")
    ax2.set_ylabel("Linear iterations")
    plt.title(title, pad=15)
    plt.savefig(file_name, dpi=600)


def run_and_report_single(Model, 
                          params: dict,
                          c_value: float, 
                          solver: str):
    
    # Run a simulation with a given nonlinear solver, and report on the number of
    # nonlinear iterations.
    # If the solver diverges to infinity, it returns 500.
    # If the solver does not converge, by exceeding the maximum number of allowed
    # Newton iterations, it returns 0.
    # If the return map solver uses more than 150 outer loop iterations, it returns -1.
    
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
        # the residual error for IRM.
        def contact_mechanics_numerical_constant(
                self, subdomains: list[pp.Grid]
        ) -> pp.ad.Scalar:
            return pp.ad.Scalar(c_value, name="Contact_mechanics_numerical_constant")
    
    if solver == "GNM":

        class Simulation(ContactMechanicsConstant,
                         SumTimeSteps,
                         Model):
            pass
    
    elif solver == "IRM":

        class Simulation(ContactMechanicsConstant,
                         SumTimeSteps,
                         ImplicitReturnMap,
                         Model):
            pass

    elif solver == "GNM-RM":

        class Simulation(ContactMechanicsConstant,
                         SumTimeSteps,
                         NewtonReturnMap,
                         Model):
            pass

    elif solver == "Delayed_GNM-RM":

        class Simulation(ContactMechanicsConstant,
                         SumTimeSteps,
                         DelayedNewtonReturnMap,
                         Model):
            pass
        
    else:
        raise NotImplementedError("Invalid nonlinear solver.")

    model = Simulation(params)
    if solver in {"GNM", "GNM-RM", "Delayed_GNM-RM"}:
        try:
            pp.run_time_dependent_model(model, params)
            itr = model.total_itr
            itr_time_step_list = model.itr_time_step
            itr_linear = sum(model.nonlinear_solver_statistics.num_krylov_iters)
            res = model.nonlinear_solver_statistics.residual_norms
            if model.params.get("make_fig4a", False):
                plt.semilogy(np.arange(0, len(res)), res, color="blue")
            elif model.params.get("make_fig4b", False):
                return_map_switch = 20
                itr = np.arange(0, len(res))
                plt.semilogy(itr[:return_map_switch+1], res[:return_map_switch+1], color="blue")
                plt.semilogy(itr[return_map_switch:], res[return_map_switch:], linestyle="--", color="blue")
            elif model.params.get("make_fig5", False):
                plt.plot(model.num_open, color="orange")
                plt.plot(model.num_stick, color="red")
                plt.plot(model.num_glide, color="blue")
        except Exception as e:
            print(e)
            itr_linear = sum(model.nonlinear_solver_statistics.num_krylov_iters)
            itr_time_step_list = model.itr_time_step
            res = model.nonlinear_solver_statistics.residual_norms
            if res[-1] > params["nl_divergence_tol"] or np.isnan(np.array(res[-1])):
                itr = "Div"
            else:
                itr = "NC"
            if model.params.get("make_fig4a", False):
                plt.semilogy(np.arange(0, len(res)), res, color="red")
            elif model.params.get("make_fig5", False):
                plt.plot(model.num_open, color="orange")
                plt.plot(model.num_stick, color="red")
                plt.plot(model.num_glide, color="blue")
    if solver == "IRM":
        try:
            run_implicit_return_map_model(model, params)
            itr = model.total_itr
            itr_linear = sum(model.nonlinear_solver_statistics.num_krylov_iters)
            itr_time_step_list = model.itr_time_step
        except Exception as e:
            print(e)
            res = model.nonlinear_solver_statistics.residual_norms
            itr_time_step_list = model.itr_time_step
            itr_linear = sum(model.nonlinear_solver_statistics.num_krylov_iters)
            if res[-1] > params["nl_divergence_tol"] or np.isnan(np.array(res[-1])):
                itr = "Div"
            elif model.outer_loop_itr > params["max_outer_iterations"]:
                itr = "NCO"
            else:
                itr = "NC"
            if model.params.get("make_fig4a", False):
                plt.semilogy(np.arange(0, len(res)), res, color="orange")
            elif model.params.get("make_fig5", False):
                plt.plot(model.num_open, color="orange")
                plt.plot(model.num_stick, color="red")
                plt.plot(model.num_glide, color="blue")
    return [itr, itr_time_step_list, itr_linear]


# Sum number of nonlinear iterations over several time steps.
class SumTimeSteps:

    def __init__(self, params):
        super().__init__(params)
        self.total_itr = 0
        self.itr_time_step = []
        self.second_phase_started = False
        self.third_phase_started = False

    def after_nonlinear_convergence(self) -> None:
        self.total_itr += self.nonlinear_solver_statistics.num_iteration
        self.itr_time_step.append(self.nonlinear_solver_statistics.num_iteration)
        # Stop the simulation if more than 505 iterations have accumulated over several time steps.
        if self.total_itr >= 505:
            raise ValueError("Simulation exceeded maximum allowed total iterations.")
        if self.time_manager.time==1*pp.HOUR and not self.second_phase_started:
            self.itr_time_step.append(-1)   # Add marker for the start of the second phase of injection.
            self.second_phase_started = True
        elif self.time_manager.time==2*pp.HOUR and not self.third_phase_started:
            self.itr_time_step.append(-2)   # Add marker for the start of the third phase of injection.
            self.third_phase_started = True
        super().after_nonlinear_convergence()

    def after_nonlinear_failure(self) -> None:
        self.total_itr += self.nonlinear_solver_statistics.num_iteration
        if self.nonlinear_solver_statistics.residual_norms[-1] > self.params["nl_divergence_tol"]:
            self.itr_time_step.append(-500)
        self.itr_time_step.append(self.nonlinear_solver_statistics.num_iteration)
        if self.total_itr >= 505:
            raise ValueError("Simulation exceeded maximum allowed total iterations.")
        super().after_nonlinear_failure()