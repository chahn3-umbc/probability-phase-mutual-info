import dfs_gaussian_class as dfs
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

p0 = 0.5
phi0 = np.pi

sigma_array = np.linspace(0.05, np.pi/2, 100)

mi_array = []

print('DFS GAUSSIAN ANALYSIS')
print('---------------------')

for sigma in sigma_array:
    ensemble = dfs.DfsGaussianMeasure(p0, phi0, sigma)
    ensemble.compute_mutual_information()
    print(f'sigma = {sigma:.4f}; mi = {ensemble.mi:.4f}')
    mi_array.append(ensemble.mi)

mi_array = np.array(mi_array)

plt.figure()
plt.plot(sigma_array, mi_array)
plt.title('Gaussian Measure - $I(P;\\Phi)$ vs $\\sigma$')
plt.ylabel('$I(P;\\Phi)$ [bits]')
plt.xlabel('$\\sigma$')
plt.grid(alpha=0.3)
plt.savefig('dfs_gaussian_analysis/dfsgaussian_mi_v_sigma.pdf')
plt.show()
