import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

start_indx = 9
end_indx = 13

delta_vals = (0.2, 0.4, 0.6, 0.8, 1.0)

results_path = './spiral_ensemble_analysis/spiral_ensemble_results/'


def linear_fit(x, d, h):
    return d * x + h


def fit_with_errors(x, y):
    """Perform linear fit and return parameters and 1-sigma errors."""
    popt, pcov = curve_fit(linear_fit, x, y)
    residuals = y - linear_fit(x, *popt)
    dof = len(x) - len(popt)
    s_sq = np.sum(residuals**2) / dof
    #pcov = pcov * s_sq
    pcov = pcov 
    perr = np.sqrt(np.diag(pcov)) * 5
    return popt, perr


gmi_arr = []
D_gmi_arr = []

gmi_err_arr = []
D_gmi_err_arr = []

for delta in delta_vals:
    df = pd.read_csv(f'{results_path}entropies_spiral_ensemble_delta={delta}.csv')
    x = np.array(df['$-\log(\epsilon)$'])
    joint_entropy = np.array(df['$H(P,\Phi)$'])
    marginal_p = np.array(df['$H(P)$'])
    marginal_phi = np.array(df['$H(\Phi)$'])

    x_arr = np.linspace(min(x), max(x), 1000)

    # Perform fits with corrected uncertainties
    popt_joint, joint_err = fit_with_errors(x[start_indx:end_indx], joint_entropy[start_indx:end_indx])
    popt_marginal_p, marginal_p_err = fit_with_errors(x[start_indx:end_indx], marginal_p[start_indx:end_indx])
    popt_marginal_phi, marginal_phi_err = fit_with_errors(x[start_indx:end_indx], marginal_phi[start_indx:end_indx])

    # Plot fits
    plt.figure(figsize=(10, 6))
    plt.plot(x, joint_entropy, 'o', label='Joint Entropy $H(P, \\Phi)$', color='blue')
    plt.plot(x_arr, linear_fit(x_arr, *popt_joint), '-', 
             label=f'Fit: $H(P, \\Phi) = {popt_joint[0]:.3f} \\cdot (-\\log(\\epsilon)) + {popt_joint[1]:.3f}$',
             color='cyan')
    plt.plot(x, marginal_p, 'o', label='Marginal Entropy $H(P)$', color='red')
    plt.plot(x_arr, linear_fit(x_arr, *popt_marginal_p), '-', 
             label=f'Fit: $H(P) = {popt_marginal_p[0]:.3f} \\cdot (-\\log(\\epsilon)) + {popt_marginal_p[1]:.3f}$',
             color='orange')
    plt.plot(x, marginal_phi, 'o', label='Marginal Entropy $H(\\Phi)$', color='green')
    plt.plot(x_arr, linear_fit(x_arr, *popt_marginal_phi), '-', 
             label=f'Fit: $H(\\Phi) = {popt_marginal_phi[0]:.3f} \\cdot (-\\log(\\epsilon)) + {popt_marginal_phi[1]:.3f}$',
             color='lime')
    plt.xlabel('$-\\log(\\epsilon)$')
    plt.ylabel('Entropy')
    plt.title(f'Entropies vs. $-\\log(\\epsilon)$ for Spiral Ensemble (delta={delta})')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'{results_path}entropy_fits_spiral_ensemble_delta={delta}.png', dpi=600)
    plt.show()

    # Extract parameters
    D_joint, H_joint = popt_joint
    D_marginal_p, H_marginal_p = popt_marginal_p
    D_marginal_phi, H_marginal_phi = popt_marginal_phi

    # Compute GMI and D_GMI
    gmi = H_marginal_p + H_marginal_phi - H_joint
    D_gmi = D_marginal_p + D_marginal_phi - D_joint

    # Propagate uncertainties (scaled to 5-sigma)
    gmi_err = 5 * np.sqrt(marginal_p_err[1]**2 + marginal_phi_err[1]**2 + joint_err[1]**2)
    D_gmi_err = 5 * np.sqrt(marginal_p_err[0]**2 + marginal_phi_err[0]**2 + joint_err[0]**2)

    print(f'Delta: {delta}, GMI: {gmi:.3f} +/- {gmi_err:.3f}, D_GMI: {D_gmi:.3f} +/- {D_gmi_err:.3f}')

    gmi_arr.append(gmi)
    D_gmi_arr.append(D_gmi)

    gmi_err_arr.append(gmi_err)
    D_gmi_err_arr.append(D_gmi_err)

gmi_arr = np.array(gmi_arr)
D_gmi_arr = np.array(D_gmi_arr)

gmi_err_arr = np.array(gmi_err_arr)
D_gmi_err_arr = np.array(D_gmi_err_arr)

plt.figure()
plt.errorbar(delta_vals, gmi_arr, yerr=gmi_err_arr, fmt='o-', capsize=4, label='$I(P;\\Phi)$', color='purple')
plt.errorbar(delta_vals, D_gmi_arr, yerr=D_gmi_err_arr, fmt='o--', capsize=4, label='$\\mathfrak{D}_I$', color='magenta')
plt.xlabel('$\\delta$')
plt.ylabel('Generalized Mutual Information')
plt.title('$I(P;\\Phi)$ and $\\mathfrak{D}_I$ vs. $\\delta$ for Spiral Ensemble')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(f'{results_path}spiral_ensemble_gmi_vs_delta_spiral_ensemble.png', dpi=600)
plt.show()
