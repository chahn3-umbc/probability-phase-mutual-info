import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from dfs_gaussian_analysis import dfs_gaussian_class as dfg


def gaussian_discrete_generation(p0, phi0, sigma, num_p_pts=10**3, num_phi_pts=10**3, csv_save_name='default.csv', plot=True):

    dp = 1/num_p_pts
    dphi = 2*np.pi/num_phi_pts

    p_values = np.linspace(0, 1, num_p_pts)
    phi_values = np.linspace(0, 2 * np.pi, num_phi_pts)

    p_mesh, phi_mesh = np.meshgrid(p_values, phi_values)

    joint_probs = dfg.DfsGaussianMeasure(p0=p0, phi0=phi0, sigma=sigma).joint_measure(p_mesh, phi_mesh)

    if plot:

        plt.figure(figsize=(10, 6))
        plt.contourf(p_values, phi_values, joint_probs, levels=100, cmap='viridis')
        plt.colorbar(label='Joint Probability Density')
        plt.xlabel('p')
        plt.ylabel('phi')
        plt.title('Joint Probability Density Function')
        plt.show()

    q_vals = []
    p_vals = []
    phi_vals = []
    for i in range(num_p_pts):
        for j in range(num_phi_pts):
            q_vals.append(joint_probs[i, j])
            p_vals.append(p_values[i])
            phi_vals.append(phi_values[j])
    q_vals = np.array(q_vals)
    p_vals = np.array(p_vals)
    phi_vals = np.array(phi_vals)

    df = pd.DataFrame({
        'p': p_vals,
        'phi': phi_vals,
        'q': q_vals
    }, columns=['p', 'phi', 'q'])

    df.to_csv(csv_save_name, index=False)
