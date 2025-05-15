import os
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

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
    
    # TODO: Document inputs to function.
    df = pd.DataFrame(data)
    df = df.astype(int)
    annot = df.astype(str).replace("0", "NC").replace("-1", "Div")
    plt.figure(figsize=(10,6))
    # cmap = sns.light_palette("#C5001A", as_cmap=True)
    # Load base colormap
    base_cmap = plt.get_cmap('YlOrRd')
    # Truncate it to avoid the darkest red
    trunc_cmap = truncate_colormap(base_cmap, 0.0, 0.85)  # 0.85 avoids deep reds
    cmap = trunc_cmap
    # cmap = sns.color_palette("YlOrRd", as_cmap=True)
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
    plt.gca().tick_params(axis="y", rotation=0)
    plt.savefig(file_name, dpi=300, bbox_inches="tight", transparent=True)

ex1_maybefinal_easy = np.array([[17,  17,   17,   17,   18,   21, 0],
[17, 17, 17, 18, 17, 17, 18],
[17, 17, 17, 17, 18, 21, 30],
[-5, 8123, 1179,  216,   78, 48, 0]]) # Uzawa       + 0.01 MPa, c=1e-2 til 1e2


ex_1_maybefinal_middle = [[36, 36, 35,  35,   0,   0,   0],
 [ 36,  35,  36,  36,  35,  37,  37],
 [ 36,  36,  35,  35,  54,  58,  54],
 [ -1,   0,   0, 787,   0,   0,   0]]

ex_1_maybefinal_hard = np.array([[0, 0, 0, 0,  0,  0,  0],
 [0, 54, 50, 81,  0, 53, 75],
 [54, 58, 57,  0, 78, 61, 61], # + 10 MPa
[0, 0, 0, 0, 0, 0, 0]])

ex_2_firstdraft = np.array([[47, 45, 46, 0, 0, 0, 0],
                           [51, 49, 50, 49, 52, 66, 63],
[0, 0, 0, 0, 0, 0, 0]])

ex1_medium_new_ref = np.array([[ 44,  44,  43,  44,   0,   0,   0],
 [ 43,  44,  43,  49,  45,  52,  45],
 [ 44,  44,  43,  44,  55,  61,  72],
 [ -1,  -1,   0, 820,   0,   0,   0]])


ex1_hard_new_ref = np.array([[  0,   0,  -1,   0,   0,   0,   0],
 [ -1,  65,  58,  49,  57,  91,  69],
 [ 82,  87,  -1,  -1,  94, 142, 126],
 [ -1,   0,  -1,  -1,   0,   0,   0]])

c_vals = ["1e-3", "1e-2", "1e-1", "1e0", "1e1", "1e2", "1e3"]
solvers = ["Newton", "NRM", "ANRM", "CRM"]
heatmap(data=ex1_hard_new_ref, vmin=10, vmax=200, xticks=c_vals, yticks=solvers,
        xlabel="c-parameter", file_name="ex1_hard_new_ref")