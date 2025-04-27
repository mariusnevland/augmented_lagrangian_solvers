import os
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


def heatmap(data: np.ndarray,
            vmin: int,
            vmax: int,
            xticks: list,
            yticks: list,
            xlabel: str,
            ylabel: str = None,
            file_name: str = None,
            title: str = None):
    
    # TODO: Document inputs to function.
    df = pd.DataFrame(data)
    df = df.astype(int)
    annot = df.astype(str).replace("0", "NC").replace("-1", "Div").replace("-5", "NCO")
    plt.figure(figsize=(10,6))
    # cmap = sns.light_palette("#C5001A", as_cmap=True)
    cmap = sns.color_palette("YlOrRd", as_cmap=True)
    cmap.set_under('#908D8D')
    # cmap.set_over('#696969')
    # cmap.set_over('green')
    mesh = sns.heatmap(df, linewidths=0.5, xticklabels=xticks,
                   yticklabels=yticks,
                   cmap=cmap,
                   cbar_kws={'label': 'Number of iterations'},
                   linecolor="black",
                   annot=annot,
                   fmt="",
                   vmin=vmin,
                   vmax=vmax,
                   annot_kws={"size": 12, "weight": "bold", "color": "black"})
    mesh.set(xlabel=xlabel, ylabel=ylabel)
    plt.savefig(file_name, dpi=300, bbox_inches="tight")


example1_vertical = np.array([[0, 32, 51, 0, 0, 0, 0],
 [17, 20, 17, 33, 19, 24, 19],
[0, 0, 0, 0, 0, 0, 0]])

example1_diagonal = np.array([[17, 13, 12,  0,  0,  0,  0],
 [14, 12, 12, 12, 12, 12, 12],
[0, 0, 0, 0, 0, 0, 0]])

c_vals = ["1e-3", "1e-2", "1e-1", "1e0", "1e1", "1e2", "1e3"]
solvers = ["Newton", "NRM", "CRM"]
heatmap(data=example1_diagonal, vmin=10, vmax=100, xticks=c_vals, yticks=solvers,
        xlabel="c-parameter", file_name="example1_diagonal")