import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from scipy.optimize import curve_fit

DATA_PATH = 'spiral_ensemble_analysis/spiral_ensemble_results/'
delta_values = (0.2, 0.4, 0.6, 0.8, 1.0)
start_indx = 9
stop_indx = 11


def linear_fit(x, a, b):
    return a * x + b


cmap = cm.get_cmap('rocket')


plt.figure(figsize=(10, 6))

colors = sns.color_palette('rocket', n_colors=5)
norm = mcolors.Normalize(vmin=min(delta_values), vmax=max(delta_values))

plt.plot(0, 0, linestyle='--', color='black', label='Linear Fit')
plt.plot(0, 0, marker='o', markersize=4, color='black', label='Data Points')

for i, color in zip(delta_values, colors):
    data = pd.read_csv(DATA_PATH+f'mutual_information_spiral_ensemble_delta={i}.csv')
    x = np.array(data['$-\log(\epsilon)$'])
    y = np.array(data['$I(P;\Phi)$'])

    popt, _ = curve_fit(linear_fit, x[start_indx:stop_indx], y[start_indx:stop_indx])

    plt.plot(x, y, marker='o', markersize=4, color=color)
    plt.plot(x, linear_fit(x, *popt), linestyle='--', color=color)
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # Needed for the colorbar

# Add colorbar
cbar = plt.colorbar(sm)
cbar.set_label('δ (Noise Level)', size=14)
plt.xlabel('$-\\log(\\epsilon)$', fontsize=14)
plt.ylabel('$I(P;\\Phi)$', fontsize=14)
plt.title('Mutual Information vs. $-\\log(\\epsilon)$ for $\\delta \\in [0.2, 1.0]$', fontsize=16)
plt.legend()
plt.grid()
plt.savefig('spiral_ensemble_analysis/spiral_ensemble_results/spiral_ensemble_MI_plot.pdf')
plt.show()
