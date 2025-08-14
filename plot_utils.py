import numpy as np
import porepy as pp
from newton_return_map import *
from implicit_return_map import *
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

"""Various utility functions for plotting results."""

# Function to truncate colormap
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=256):
    new_cmap = LinearSegmentedColormap.from_list(
        f"trunc({cmap.name},{minval:.2f},{maxval:.2f})",
        cmap(np.linspace(minval, maxval, n))
    )
    return new_cmap


def heatmap(data: np.ndarray,
            vmin: int,
            vmax: int,
            xticks: list,
            yticks: list,
            xlabel: str,
            ylabel: str = None,
            file_name: str = None,
            title: str = None):
    
    """Create a heatmap visualizing the number of iterations."""
    df = pd.DataFrame(data)
    df = df.astype(int)
    annot = df.astype(str).replace("0", "NC").replace("500", "Div").replace("-1", "NCO")
    try:
        # Trick to avoid needing to display a larger color map range due to outliers.
        if annot[4][2] == "100":
            annot[4][2] = "291"
        elif annot[3][2] == "100":
            annot[3][2] = "360"
    except:
        pass
    plt.figure(figsize=(10,6))
    # Load base colormap
    base_cmap = plt.get_cmap('YlOrRd')
    # Truncate it to avoid the darkest red
    trunc_cmap = truncate_colormap(base_cmap, 0.0, 0.85)  # 0.85 avoids deep reds
    cmap = trunc_cmap
    # Non-convergent cases get assigned a grey color.
    cmap.set_under('#908D8D')
    cmap.set_over('#696969')
    mesh = sns.heatmap(df, linewidths=0.5, xticklabels=xticks,
                   yticklabels=yticks,
                   cmap=cmap,
                   linecolor="black",
                   annot=annot,
                   fmt="",
                   vmin=vmin,
                   vmax=vmax,
                   annot_kws={"size": 16, "weight": "bold", "color": "black"})
    mesh.set(xlabel=xlabel, ylabel=ylabel)
    cbar = mesh.collections[0].colorbar
    cbar.set_label("Number of iterations", fontsize=16)
    cbar.ax.tick_params(labelsize=14)
    plt.xlabel(xlabel, fontsize=16)
    plt.gca().tick_params(axis="y", rotation=0)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title(title, fontsize=20)
    plt.savefig(file_name, dpi=300, bbox_inches="tight", transparent=True)


def run_and_report_single(Model, 
                          params: dict,
                          c_value: float, 
                          solver: str) -> int:
    
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
        except:
            res = model.nonlinear_solver_statistics.residual_norms
            if res[-1] > 1e5 or np.isnan(np.array(res[-1])):
                itr = 500
            else:
                itr = 0
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
        except:
            res = model.nonlinear_solver_statistics.residual_norms
            if res[-1] > 1e5 or np.isnan(np.array(res[-1])):
                itr = 500
            elif model.outer_loop_itr > 150:
                itr = -1
            else:
                itr = 0
            if model.params.get("make_fig4a", False):
                plt.semilogy(np.arange(0, len(res)), res, color="orange")
            elif model.params.get("make_fig5", False):
                plt.plot(model.num_open, color="orange")
                plt.plot(model.num_stick, color="red")
                plt.plot(model.num_glide, color="blue")
    return itr


# Sum number of nonlinear iterations over several time steps.
class SumTimeSteps:

    total_itr = 0  # Total number of iterations across all time steps.

    def after_nonlinear_convergence(self) -> None:
        super().after_nonlinear_convergence()
        self.total_itr += self.nonlinear_solver_statistics.num_iteration