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
    annot = df.astype(str).replace("0", "NC").replace("500", "Div").replace("-1", "NCO")
    if annot[4][2] == "100":
        annot[4][2] = "317"
    elif annot[3][2] == "100":
        annot[3][2] = "360"
    plt.figure(figsize=(10,6))
    # cmap = sns.light_palette("#C5001A", as_cmap=True)
    # Load base colormap
    base_cmap = plt.get_cmap('YlOrRd')
    # Truncate it to avoid the darkest red
    trunc_cmap = truncate_colormap(base_cmap, 0.0, 0.85)  # 0.85 avoids deep reds
    cmap = trunc_cmap
    # cmap = sns.color_palette("YlOrRd", as_cmap=True)
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

ex1_easy = np.array([[6 ,6,  6,  6,  6,  6,  6,  0],
 [8, 7,  9,  10,  8,  9,  8,  9],
[-1, -1, -1, 100,  61,  22,  14,   0]])  # Need to run the two lowest c-parameters

ex1_easy_new_ref_one_time = np.array([[5,  5,  5,  5,  5,  7,  0],
 [ 5,  5,  5,  5,  5,  6,  5],
[14854, 2182,  292,   53,   20,  13,    0]])

ex1_medium = np.array([[9, 9,  9,  9,  9,  0,  0,  0],
 [12, 11, 14, 12, 14, 12, 11, 10],  # +1MPa, enda finere grid
 [500, 500, 0, 0, 100, 0, 0, 0]])   # +1MPa, CRM 100 is really 317


ex1_hard = np.array([[0, 500, 500,  25, 500, 500, 500, 500],
[500, 26, 46, 32, 28, 31, 43, 23],
[500, 500, 500, 500, 500, 0, 0, 500]])    # +8MPa, enda finere grid

ex1_grid = np.array([[51, 53, 58, 58],
                    [62, 69, 72, 79],
                    [155, 500, 500, 156],
                    [99, 111, 106, 128]])

ex1_ultrahard = np.array([[0, 500,  77,  19, 500, 0, 0, 500],   # +10MPa, enda finere grid
[500, 500, 21, 47, 24, 44, 25, 34],
[500, 500, 500, 500, 500, 0, 500, 0]])

ex2_alt = np.array([[15, 15, 15, 500,  0,  0, 0],
 [35, 45, 46, 43, 53, 45, 92], # +5MPa, new frac network, old coarse grid, 1e-3 til 1e3
[15, 15, 15, 500, 91, 67, 57]]) 

ex2 = np.array([[15, 14, 15, 500, 0, 0, 0],
[47, 26, 37, 27, 27, 52, 33],
[0, 21, 17, 500, 0, 0, 0],
[84, 129, 108, 100, 91, 86, 86]])

ex1_medium_presentation = np.array([[ 10,  10,  10,  10,   0,   0,   0],
 [ 10,  11,  12,  12,  12,  12,  10],
 [ 500,  500,   0, 100,   0,   0,   0]]) # +1MPa

ex1_easy_presentation = np.array([[6,6,6,6,6,6,0],
                                  [7,9,10,8,9,8,9],
                                  [16037,2423,360,61,22,14,0]])


ex1_hard_presentation = np.array([[ 0,  0,  0, 500,  0, 0,  0],
[29, 45, 19, 16, 15, 22, 14],
[ 0,  0, 500,  0,  0,  0,  0]])

ex1_low_dil = np.array([[500,  56, 500, 500, 500,   0,   0,   0],
 [500, 500, 500,  47, 500,  20,  55,  23],
 [500, 500,   0, 500, 500, 500, 500, 500]])

ex1_high_dil = np.array([[500,  69,  53,  22, 500, 500,   0, 500],
 [500,  28,  28,  23,  21,  22,  93,  29],
 [500, 500, 500,   0, 500,   0,   0, 500]])

c_vals = ["1e-4", "1e-3", "1e-2", "1e-1", "1e0", "1e1", "1e2", "1e3"]
solvers = ["GNM", "GNM-RM", "RM"]
xticks_grid = ["33830", "64068", "121501", "199282"]
yticks_grid = ["Newton, well pressure 21MPa, c=1e-1", "NRM, well pressure 21MPa, c=1e-1",
               "Newton, well pressure 30MPa, c=1e-2", "NRM, well pressure 30MPa, c=1e-2"]
yticks_3D = ["Newton, well pressure 25MPa", "NRM, well pressure 25MPa",
               "Newton, well pressure 30MPa", "NRM, well pressure 30MPa"]
xlabel_grid = "Number of cells"
heatmap(data=ex1_high_dil, vmin=1, vmax=100, xticks=c_vals, yticks=solvers,
        xlabel="c-parameter [GPa/m]", file_name="ex1_high_dil", title=r"Injection pressure 30MPa, $\psi$=7 degrees")