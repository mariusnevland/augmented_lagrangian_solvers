import numpy as np
import porepy as pp
from newton_return_map import *
from implicit_return_map import *
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

"""Various utility functions for plotting results."""

def bar_chart(itr_time_step_list, lin_list, ymin, ymax, num_c_values, labels, file_name):
    width = 0.1
    indices = ['A', 'B', 'C']
    positions = [i * 0.5 for i in range(num_c_values)]
    positions2 = [-1, 0, 1]
    colors = ['#1f77b4', '#9f1b1b', '#2ca02c']
    _, ax1 = plt.subplots()
    ax1.set_ylim(ymin, ymax)
    _, ymax = ax1.get_ylim()
    ax2 = ax1.twinx()
    for (i, pos) in enumerate(positions):
        df = pd.DataFrame(itr_time_step_list[i], index=indices)
        for (ind, col, lin, foo) in zip(indices, colors, lin_list[i], positions2):
            bottom = 0
            for value in df.loc[ind].dropna():
                if value==31:
                    bar = ax1.bar(pos + foo * width, value, width, align='center', bottom=bottom, edgecolor='black', hatch='/', linewidth=0.5, color=col)
                else:
                    bar = ax1.bar(pos + foo * width, value, width, align='center', bottom=bottom, edgecolor='black', linewidth=0.5, color=col)
                bottom += value
                if bottom > ymax:
                    rect = bar[0]
                    xmid = rect.get_x() + rect.get_width() / 2
                    ax1.plot(xmid, ymax * 1.0, marker=(3,0,0), markersize=10, color='red')
                    break
            ax2.plot(pos + foo * width, lin, marker='o', color='black')
            special_bar = plt.bar(pos + foo * width, 5, width, align='center', bottom=bottom, color='none', edgecolor='none')[0]
            ax = plt.gca()
            x_center = special_bar.get_x() + special_bar.get_width() / 2
            y_center = special_bar.get_y() + special_bar.get_height() / 2
    # Set the xtick at the center bar (positions[1]) so the tick is in the middle of the 2nd bar
    plt.xticks(positions, labels)
    ax1.set_xlabel("c-value [GPa/m]")
    ax1.set_ylabel("Nonlinear iterations")
    ax2.set_ylabel("Linear iterations")
    plt.title("Nonlinearity: No aperture")
    plt.savefig(file_name)


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
        outer_itr = 0
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
            outer_itr = model.accumulated_outer_loop_itr
        except:
            res = model.nonlinear_solver_statistics.residual_norms
            itr_time_step_list = model.itr_time_step
            itr_linear = sum(model.nonlinear_solver_statistics.num_krylov_iters)
            outer_itr = model.accumulated_outer_loop_itr
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

    def after_nonlinear_convergence(self) -> None:
        self.total_itr += self.nonlinear_solver_statistics.num_iteration
        self.itr_time_step.append(self.nonlinear_solver_statistics.num_iteration)
        super().after_nonlinear_convergence()

    def after_nonlinear_failure(self) -> None:
        self.total_itr += self.nonlinear_solver_statistics.num_iteration
        self.itr_time_step.append(self.nonlinear_solver_statistics.num_iteration)
        super().after_nonlinear_failure()