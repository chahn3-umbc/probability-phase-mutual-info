import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
import pandas as pd
from itertools import product


def to_spherical(p_vals, phi_vals):
    theta_vals = 2 * np.arccos(np.sqrt(1 - p_vals))
    return theta_vals, phi_vals


def to_cartesian(p_vals, phi_vals):
    theta, phi = to_spherical(p_vals, phi_vals)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return x, y, z


def calc_entropy(prob_dist):
    valid_probs = prob_dist[prob_dist > 0]
    entropy = -np.sum(valid_probs * np.log2(valid_probs))
    return entropy


def von_neumann_entropy(rho):
    vals = np.linalg.eigvalsh(rho)
    vals = vals[vals > 0]
    return -np.sum(vals * np.log2(vals))


def linear_fit(x, m, b):
    return m * x + b


def get_projected_ensemble(psi, L, site_A):
    """
    Compute the projected ensemble for a single subsystem qubit (site_A)
    by measuring all other qubits (subsystem_B) in the z-basis.
    
    Following the construction in Colter et al. (PRX Quantum 4, 010311, 2023):
        - psi: full many-body state as a 1D numpy array of length 2^L (complex)
        - L: total number of qubits in the system
        - site_A: index of the subsystem qubit (0 <= site_A < L)
        
    Returns:
        ensemble: list of (prob, state_A) tuples where
            - prob is a float p(z_B)
            - state_A is a 2D numpy array representing the post-measurement state of qubit A
    """

    psi_tensor = psi.reshape([2] * L)  # Reshape to L qubits

    B_sites = [i for i in range(L) if i != site_A]

    ensemble = []

    for z_B in product([0, 1], repeat=len(B_sites)):

        idx = [None] * L
        for b_pos, b_val in zip(B_sites, z_B):
            idx[b_pos] = b_val
        idx[site_A] = slice(None)  # Keep site_A as a slice to
        idx = tuple(idx)

        psi_A_unnorm = psi_tensor[idx]  # Shape (2,)
        prob = float(np.real(np.dot(psi_A_unnorm.conj(), psi_A_unnorm)))  # p(z_B) = ||psi_A_unnorm||^2

        if prob < 1e-14:
            continue

        psi_A = psi_A_unnorm / np.sqrt(prob)  # Normalize to get the post-measurement state of A

        ensemble.append((prob, psi_A))

    total_prob = sum(p for p, _ in ensemble)
    assert abs(total_prob - 1) < 1e-8, f"Total probability {total_prob} does not sum to 1"

    return ensemble


def get_p_phi(ensemble):

    q_vals = []
    p_vals = []
    phi_vals = []

    for entry in ensemble:
        prob, psi_A = entry
        p = np.abs(psi_A[1])**2
        phi = np.angle(psi_A[1]) - np.angle(psi_A[0])
        phi = phi % (2 * np.pi)  # Ensure phi is in [0, 2pi)
        # print(f"p: {p:.4f}, phi: {phi:.4f}")

        q_vals.append(prob)
        p_vals.append(p)
        phi_vals.append(phi)

    return q_vals, p_vals, phi_vals


class Ensemble:

    def __init__(self, q_vals, p_vals, phi_vals, ensemble_name='default', cells_per_dim=(1, 4, 16, 64, 256, 1024, 4096, 16384)):
        self.q = q_vals
        self.p = p_vals
        self.phi = phi_vals

        self.starting_fit_point = None
        self.ending_fit_point = None

        self.name = ensemble_name

        self.geometric_mutual_info_dim = None
        self.geometric_mutual_info = None

        self.rho = self.calc_density_matrix()

        self.cells_per_dim = cells_per_dim
        self.joint_hists = []
        self.p_hists = []
        self.phi_hists = []
        self.x_edges = []
        self.y_edges = []
        self.p_ent = np.zeros(len(self.cells_per_dim))
        self.phi_ent = np.zeros(len(self.cells_per_dim))
        self.joint_ent = np.zeros(len(self.cells_per_dim))
        self.mutual_info = np.zeros(len(self.cells_per_dim))
        self.geometric_ent = []
        self.geometric_information_dim = []

    def plot_phase_space(self):
        plt.figure(figsize=(8, 6))
        plt.scatter(self.p, self.phi, c=self.q)
        plt.xlabel('Probabilities $p$')
        plt.ylabel('Phases $\\phi$')
        plt.title(f'Phase Space - {self.name.split("_")[0]} Ensemble')
        plt.xlim(0, 1)
        plt.ylim(0, 2 * np.pi)
        plt.colorbar()
        plt.savefig(f'deep_thermalization_analysis/ED_analysis_results/{self.name}/phase_space_plot_{self.name}.png', dpi=600)
        plt.close()

    def build_directory(self):
        try:
            os.makedirs(f'deep_thermalization_analysis/ED_analysis_results/{self.name}')
            print(f'Directory \'deep_thermalization_analysis/ED_analysis_results/{self.name}\' created successfully.')
        except FileExistsError:
            print(f'Directory \'deep_thermalization_analysis/ED_analysis_results/{self.name}\' already exists.')
        except Exception as e:
            print(f'An error occurred: {e}')

    def analyze(self, plot=True):

        self.build_directory()
        if plot:
            self.plot_phase_space()
        else:
            pass

        self.process_joint_distributions()
        self.process_marginal_distributions()

        self.process_entropy(self.joint_ent, 'joint', plot=plot)
        self.process_entropy(self.p_ent, 'p', plot=plot)
        self.process_entropy(self.phi_ent, 'phi', plot=plot)
        self.process_mutual_info(plot=plot)

        print('\nFinal Results:')
        print(f'Ensemble: {self.name}')
        print(f'Number of points: {len(self.p)}')
        print(f'Geometric Mutual Information: {self.geometric_mutual_info:.6f}')
        print(f'Information Dimension: {self.geometric_mutual_info_dim:.6f}')

    def process_joint_distributions(self):

        for i in range(len(self.cells_per_dim)):
            hist, x_edges, y_edges = np.histogram2d(
                self.p, self.phi,
                bins=self.cells_per_dim[i],
                range=[[0, 1], [0, 2*np.pi]],
                weights=self.q
            )

            hist = hist / np.sum(hist)

            self.joint_hists.append(hist.T)
            self.x_edges.append(x_edges)
            self.y_edges.append(y_edges)

            self.joint_ent[i] = calc_entropy(hist.flatten())

    def process_marginal_distributions(self):

        for i in range(len(self.joint_hists)):
            p_hist = np.sum(self.joint_hists[i], axis=0)
            phi_hist = np.sum(self.joint_hists[i], axis=1)

            self.p_hists.append(p_hist)
            self.phi_hists.append(phi_hist)

            self.p_ent[i] = calc_entropy(p_hist)
            self.phi_ent[i] = calc_entropy(phi_hist)

    def process_mutual_info(self, plot=True):
        x = np.log2(self.cells_per_dim)

        for i in range(len(self.cells_per_dim)):
            mi = self.p_ent[i] + self.phi_ent[i] - self.joint_ent[i]
            self.mutual_info[i] = max(0, mi)

        if self.ending_fit_point > self.starting_fit_point:
            popt, pcov = curve_fit(linear_fit, x[self.starting_fit_point:self.ending_fit_point + 1],
                                   self.mutual_info[self.starting_fit_point:self.ending_fit_point + 1])

            if plot:
                x_array = np.linspace(0, max(x), 10000)
                plt.figure(figsize=(10, 6))
                plt.plot(x, self.mutual_info, marker='o', label='Data')
                plt.plot(x_array, linear_fit(x_array, popt[0], popt[1]),
                         label=f'Fit: {popt[0]:.4f}*log(L) + {popt[1]:.4f}')
                plt.axvspan(x[self.starting_fit_point], x[self.ending_fit_point], alpha=0.2, color='green', label='Fitting Region')
                plt.xlabel('$\\log(L) = -\\log({\\epsilon})$')
                plt.ylabel('Mutual Information $I(P;\\Phi)$ [bits]')
                plt.title(f'Mutual Information $I(P;\\Phi)$ - {self.name} Ensemble')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.savefig(f'deep_thermalization_analysis/ED_analysis_results/{self.name}/mutual_info_plot_{self.name}.png', dpi=600)
                plt.close()
            else:
                pass

            self.geometric_mutual_info = popt[1]
            self.geometric_mutual_info_dim = popt[0]

    def process_entropy(self, entropy, name, plot=True):
        x = np.log2(self.cells_per_dim)

        fit_points = []
        fit_points = np.array(fit_points)

        p0 = np.min(fit_points) if len(fit_points) > 0 else 3
        p1 = np.max(fit_points) if len(fit_points) > 0 else 5

        if name == 'joint':
            self.starting_fit_point = p0
            self.ending_fit_point = p1

        if p1 > p0 + 1:
            popt, pcov = curve_fit(linear_fit, x[p0:p1 + 1], entropy[p0:p1 + 1])

            x_array = np.linspace(0, max(x), 10000)

            if plot:
                plt.figure(figsize=(10, 6))
                plt.plot(x, entropy, marker='o', label='Data')
                plt.plot(x_array, linear_fit(x_array, popt[0], popt[1]),
                         label=f'Fit: {popt[0]:.4f}*log(L) + {popt[1]:.4f}')
                plt.axhline(y=calc_entropy(np.array(self.q)),
                            color='r', linestyle='--', label='Sample Size Limit')
                plt.axvspan(x[p0], x[p1], alpha=0.2, color='green', label='Fitting Region')
                plt.xlabel('$\\log(L) = -\\log({\\epsilon})$')
                plt.ylabel('Entropy $H$ [bits]')

                title_map = {
                    'joint': f'Joint Entropy $H(p,\\phi)$ - {self.name} Ensemble',
                    'p': f'Marginal Entropy $H(p)$ - {self.name} Ensemble',
                    'phi': f'Marginal Entropy $H(\\phi)$ - {self.name} Ensemble'
                }
                plt.title(title_map.get(name, f'Entropy - {name}'))

                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.savefig(f'deep_thermalization_analysis/ED_analysis_results/{self.name}/entropy_plot_{name}_{self.name}.pdf')
                plt.close()
            else:
                pass

            self.geometric_ent.append(popt[1])
            self.geometric_information_dim.append(popt[0])

    def calc_density_matrix(self):
        rho_00 = np.average(1 - self.p)
        rho_01 = np.average(np.sqrt(self.p * (1 - self.p)) * np.exp(-1j * self.phi))
        rho_10 = np.average(np.sqrt(self.p * (1 - self.p)) * np.exp(1j * self.phi))
        rho_11 = np.average(self.p)

        rho = np.array([[rho_00, rho_01],
                        [rho_10, rho_11]])
        return rho

    def relative_entropy_of_coherence(self):
        if self.rho is None:
            self.calc_density_matrix()
        rho_diag = np.diag(np.diag(self.rho))
        return von_neumann_entropy(rho_diag) - von_neumann_entropy(self.rho)

    def save_ent(self):
        headers = [f'$-\\log(\\epsilon)$', f'Results']

        joint_data_frame = pd.DataFrame(np.array([np.log2(self.cells_per_dim), self.joint_ent]).T, columns=headers)
        p_data_frame = pd.DataFrame(np.array([np.log2(self.cells_per_dim), self.p_ent]).T, columns=headers)
        phi_data_frame = pd.DataFrame(np.array([np.log2(self.cells_per_dim), self.phi_ent]).T, columns=headers)

        mutual_info_frame = pd.DataFrame(np.array([np.log2(self.cells_per_dim), self.mutual_info]).T, columns=headers)

        joint_data_frame.to_csv(f'deep_thermalization_analysis/ED_analysis_results/{self.name}/joint_ent_data.csv', header=True, index=False)
        p_data_frame.to_csv(f'deep_thermalization_analysis/ED_analysis_results/{self.name}/p_ent_data.csv', header=True, index=False)
        phi_data_frame.to_csv(f'deep_thermalization_analysis/ED_analysis_results/{self.name}/phi_ent_data.csv', header=True, index=False)

        mutual_info_frame.to_csv(f'deep_thermalization_analysis/ED_analysis_results/{self.name}/mutual_info_data.csv', header=True, index=False)