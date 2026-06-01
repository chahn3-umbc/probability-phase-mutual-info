import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import seaborn as sns

data = pd.read_csv('canonical_ensemble_analysis/mutual_information_v_beta_g.csv', index_col=0)

plt.figure(figsize=(12, 8))
mi_matrix = data.to_numpy()

plt.imshow(mi_matrix, origin='lower', cmap='rocket')
plt.colorbar().set_label(label='Mutual Information $I(P;\\Phi)$', size=20)

beta_vals = np.linspace(0.01, 100, 51)
g_vals = np.linspace(0, 2, 51)

beta_ticks = np.arange(0, len(beta_vals), 10)
g_ticks = np.arange(0, len(g_vals), 5)

mpl.rcParams.update({'font.size': 22})
mpl.rc('xtick', labelsize=20)
mpl.rc('ytick', labelsize=20)

#plt.xticks(beta_ticks, [f'{beta_vals[i]:.2f}' for i in beta_ticks], fontsize=14)
plt.yticks(g_ticks, [f'{g_vals[i]:.2f}' for i in g_ticks], fontsize=14)
plt.xticks([0, 12.5, 25, 37.5, 50], ['0.01', '25', '50', '75', '100'], fontsize=14)

plt.xlabel('Inverse Temperature $\\beta$', fontsize=20)
plt.ylabel('Coupling Strength $g$', fontsize=20)
plt.title('Mutual Information $I(P;\\Phi)$ Heatmap', fontsize=20)
plt.savefig('canonical_ensemble_analysis/mutual_information_v_beta_g.png')
plt.show()

results = {}

for i in range(len(g_vals)-1):
    mi_array = []
    for j in range(len(beta_vals)-1):
        mi_array.append(mi_matrix[i,j])
    results[g_vals[i]] = mi_array

print(mi_matrix)
