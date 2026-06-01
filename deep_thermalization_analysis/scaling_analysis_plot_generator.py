import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

N_vals = [8, 10, 12, 14]
markers = ['o', 's', 'D', '^']

plt.figure()
for i, n in enumerate(N_vals):
    df = pd.read_csv(f'deep_thermalization_analysis/processed_data/mutual_info_data_N={n}.csv')
    x = df['$-\log(\epsilon)$']
    y = df['Results']

    plt.plot(x, y, label=f'$N={n}$', marker=markers[i], linestyle='-', markersize=8)
plt.xlabel(r'$-\log(\epsilon)$')
plt.ylabel(r'$I(P;\Phi)$')
plt.title(r'Mutual Information vs. $-\log(\epsilon)$ for Different N')
plt.axvspan(0, 6, alpha=0.1, color='blue') # , label=r'$\mathfrak{D}_I = 0$ Region'
plt.axvspan(6, 10, alpha=0.1, color='orange') # , label='Unstable region'
plt.axvspan(10, 14, alpha=0.1, color='red') # , label='Finite size effects'
plt.legend()
plt.show()
