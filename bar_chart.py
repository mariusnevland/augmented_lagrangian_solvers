import pandas as pd
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import numpy as np

itr_time_step_list = [[6, 4, 4, 4, 5, 5, 4, 5, 4, 5, 4, 4, 1], [7, 3, 5, 4, 6, 4, 4, 5, 4, 5, 4, 4, 1]]
itr_list = [55, 56]

df = pd.DataFrame(itr_time_step_list, index=['A', 'B'])

x_pos = 0
width = 0.2

# First stacked bar
bottom = 0
for value in df.loc['A']:
    if value==31:
        plt.bar(x_pos - width/2, value, width, bottom=bottom, edgecolor='black', color='grey')
    else:
        plt.bar(x_pos - width/2, value, width, bottom=bottom, edgecolor='black', color='#1f77b4')
    bottom += value
# invisible container bar (same size and position) -- can be used for clipping/alignment
special_bar = plt.bar(x_pos + width/2, 5, width, bottom=bottom, color='none', edgecolor='none')[0]
# place centered text inside the special bar
ax = plt.gca()
x_center = special_bar.get_x() + special_bar.get_width() / 2
y_center = special_bar.get_y() + special_bar.get_height() / 2
ax.text(x_center, y_center, 'Special', ha='center', va='center', color='black', fontsize=9, fontweight='bold')

# Second stacked bar
bottom = 0
for value in df.loc['B']:
    if value==31:
        plt.bar(x_pos + width/2, value, width, bottom=bottom, edgecolor='black', color='grey')
    else:
        plt.bar(x_pos + width/2, value, width, bottom=bottom, edgecolor='black', color="#9f1b1b")
    bottom += value

plt.xticks([x_pos], ['X'])
plt.ylabel("Nonlinear iterations")
plt.title("Stacked bars from DataFrame")
plt.savefig("bar_test.png")