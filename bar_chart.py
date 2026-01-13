import pandas as pd
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import numpy as np



it_list_1 = [[6, 4, 4, 4, 5, 5, 4, 5, 4, 5, 4, 4, 1], [7, 3, 5, 4, 6, 4, 4, 5, 4, 5, 4, 4, 1],[5, 5, 4, 5, 6, 7, 3, 1, 4, 4, 4, 4, 5, 5, 4, 1, 3, 4, 4, 4, 4, 5, 2, 1, 3, 4, 4, 5, 4, 4, 2, 1, 4, 5, 4, 5, 5, 4, 4, 1, 4, 5, 4, 5, 4, 4, 4, 1, 4, 5, 4, 5, 5, 5, 3, 1, 1, 1, 1]]
it_list_2 = [[6, 5, 4, 3, 5, 4, 4, 5, 4, 5, 1, 1, 1], [9, 4, 5, 2, 5, 2, 6, 3, 6, 3, 5, 1, 1, 1], [4, 4, 5, 6, 7, 3, 1, 4, 4, 4, 5, 5, 3, 1, 4, 4, 4, 5, 5, 2, 1, 3, 4, 5, 5, 5, 4, 1, 4, 4, 5, 4, 4, 4, 1, 4, 4, 5, 5, 5, 4, 1, 4, 5, 5, 5, 5, 3, 1, 1, 1]]
it_list_3 = [[6, 5, 4, 3, 5, 4, 5, 4, 4, 5, 5, 1, 1, 1], [9, 3, 5, 2, 6, 4, 4, 3, 8, 1, 1, 1, 1], [5, 5, 6, 6, 3, 1, 4, 4, 5, 5, 4, 1, 4, 4, 4, 5, 2, 1, 3, 4, 4, 4, 2, 1, 4, 3, 5, 4, 3, 1, 4, 3, 4, 4, 3, 1, 4, 4, 5, 5, 3, 1, 1, 1, 1]]
it_list_4 = [[6, 5, 4, 4, 5, 4, 5, 4, 4, 5, 1, 1, 1, 1], [8, 4, 5, 2, 7, 4, 5, 3, 6, 4, 6, 1, 1, 1], [6, 6, 6, 3, 1, 4, 4, 6, 4, 1, 3, 3, 5, 2, 1, 3, 3, 4, 2, 1, 3, 3, 4, 4, 1, 3, 3, 4, 3, 1, 4, 4, 5, 3, 1, 1, 1, 1]]
it_list_5 = [[8, 5, 4, 4, 5, 4, 4, 5, 5, 4, 5, 3, 1, 1], [10, 4, 5, 2, 6, 4, 4, 4, 5, 1, 1, 1, 1], [8, 7, 6, 2, 1, 4, 3, 5, 1, 3, 3, 3, 1, 3, 3, 2, 1, 3, 3, 4, 1, 3, 3, 4, 1, 3, 7, 4, 1, 5, 4, 4, 1, 1, 1]]
it_list_6 = [[31, 31, 31, 31, 31], [10, 5, 4, 4, 2, 6, 3, 8, 3, 6, 1, 1, 1], [31, 31, 31, 31, 31]]
itr_time_step_list = [it_list_1, it_list_2, it_list_3, it_list_4, it_list_5, it_list_6]

# itr_list_1 = [55, 56, 219]
# itr_list_2 = [48, 53, 192]
# itr_list_3 = [53, 48, 151]
# itr_list_4 = [50, 57, 117]
# itr_list_5 = [58, 48, 109]
# itr_list_6 = ['NC', 54, 'NC']
# itr_list = [itr_list_1, itr_list_2, itr_list_3, itr_list_4, itr_list_5, itr_list_6]


x_pos = 0
width = 0.1

# First stacked bar (three bars clustered around the central tick)
indices = ['A', 'B', 'C']
positions = [0, 0.5, 1, 1.5, 2, 2.5]
positions2 = [-1, 0, 1]
# positions = [x_pos - width, x_pos, x_pos + width]
colors = ['#1f77b4', '#9f1b1b', '#2ca02c']
for (i, pos) in enumerate(positions):
    df = pd.DataFrame(itr_time_step_list[i], index=indices)
    for (ind, col, foo) in zip(indices, colors, positions2):
        bottom = 0
        for value in df.loc[ind].dropna():
            if value==31:
                plt.bar(pos + foo * width, value, width, align='center', bottom=bottom, edgecolor='black', hatch='/', linewidth=0.5, color=col)
            else:
                plt.bar(pos + foo * width, value, width, align='center', bottom=bottom, edgecolor='black', linewidth=0.5, color=col)
            bottom += value
        special_bar = plt.bar(pos + foo * width, 5, width, align='center', bottom=bottom, color='none', edgecolor='none')[0]
        ax = plt.gca()
        x_center = special_bar.get_x() + special_bar.get_width() / 2
        y_center = special_bar.get_y() + special_bar.get_height() / 2
# Set the xtick at the center bar (positions[1]) so the tick is in the middle of the 2nd bar
plt.xticks(positions, ['1e-3', '1e-2', '1e-1', '1e0', '1e1', '1e2'])
plt.ylim(0,200)
plt.xlabel("c-value [GPa/m]")
plt.ylabel("Nonlinear iterations")
plt.title("Nonlinearity: No aperture")
plt.savefig("bar_test.png")