import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
import pandas as pd
import ensemble_scaling_analysis.comp_functions as cf

class EnsembleAnalysis:
    def __init__(self, ensemble_name='default', cells_per_dim=np.array([2**i for i in range(0, 15)]), 
                 data_file='data.csv', results_dir='results', fit_points=(3, 5)):
        self.ensemble_name = ensemble_name
        self.cells_per_dim = cells_per_dim
        self.data_file = data_file
        self.results_dir = results_dir

        self.q, self.p, self.phi = self.load_ensemble_data()

        self.joint_ent = np.zeros(len(self.cells_per_dim))
        self.p_ent = np.zeros(len(self.cells_per_dim))
        self.phi_ent = np.zeros(len(self.cells_per_dim))
        self.mi = np.zeros(len(self.cells_per_dim))

        self.joint_hists = []

        self.start_fit_index = np.min(fit_points)
        self.end_fit_index = np.max(fit_points) + 1

        self.gmi = None
        self.gmi_dim = None

    def load_ensemble_data(self):
        # Implement logic to load ensemble data
        ensemble_df = pd.read_csv(self.data_file)
        return ensemble_df['q'].values, ensemble_df['p'].values, ensemble_df['phi'].values
    
    def plot_phase_space(self):
        plt.figure(figsize=(10, 6))
        plt.scatter(self.p, self.phi, c=self.q, cmap='viridis', s=10)
        plt.colorbar(label='$q$')
        plt.xlabel('Probabilities $p$')
        plt.ylabel('Phases $\\phi$')
        plt.title(f'Phase Space - {self.ensemble_name} Ensemble')
        plt.savefig(os.path.join(self.results_dir, f'phase_space_{self.ensemble_name}.png'))
        plt.close()

    def build_directory_structure(self):
        os.makedirs(self.results_dir, exist_ok=True)

    def analyze(self, plot_type=True, save_status=True):
        # Perform analysis on the data
        self.build_directory_structure()
        if plot_type:
            self.plot_phase_space()
        else:
            pass  # Implement other plot types as needed

        for cell_count in self.cells_per_dim:
            self.analyze_cell_count(cell_count)

        self.process_mutual_information()

        print('\nFinal Results:')
        print(f'Ensemble               : {self.ensemble_name}')
        print(f'Number of points       : {len(self.q)}')
        print(f'Geometric MI           : {self.gmi:.4f} bits')
        print(f'Geometric MI Dimension : {self.gmi_dim:.4f}')

        if save_status:
            self.save_ent()
            self.save_mi()

    def analyze_cell_count(self, cell_count):
        # Implement analysis for each cell count
        self.process_joint_distribution(cell_count)
        self.process_marginal_distributions(cell_count)

    def process_joint_distribution(self, cell_count):
        # Implement joint distribution processing
        hist, x_edges, y_edges = np.histogram2d(
            self.p, self.phi, 
            bins=cell_count, range=[[0, 1], [0, 2 * np.pi]], 
            weights=self.q
        )
        hist = hist / np.sum(hist)  # Normalize

        self.joint_hists.append(hist.T)

        indx = np.where(self.cells_per_dim == cell_count)[0]

        # Calculate entropies
        self.joint_ent[indx] = cf.calc_entropy(hist.flatten())

    def process_marginal_distributions(self, cell_count):

        indx = int(np.where(self.cells_per_dim == cell_count)[0][0])

        # Implement marginal distributions processing
        p_hist = np.sum(self.joint_hists[indx], axis=0)
        phi_hist = np.sum(self.joint_hists[indx], axis=1)

        self.p_ent[indx] = cf.calc_entropy(p_hist)
        self.phi_ent[indx] = cf.calc_entropy(phi_hist)

    def process_mutual_information(self, plot=True):
        self.mi = self.p_ent + self.phi_ent - self.joint_ent

        if self.end_fit_index > self.start_fit_index:
            fit_cells = self.cells_per_dim[self.start_fit_index:self.end_fit_index]
            fit_mi = self.mi[self.start_fit_index:self.end_fit_index]

            popt, _ = curve_fit(cf.linear_fit, np.log2(fit_cells), fit_mi)
            m, b = popt

            if plot:
                self.plot_mutual_information(m, b)

        self.gmi = b
        self.gmi_dim = m

    def plot_mutual_information(self, m, b):
        smooth_x = np.linspace(np.log2(self.cells_per_dim[0]), np.log2(self.cells_per_dim[-1]), 1000)
        plt.figure(figsize=(10, 6))
        plt.plot(np.log2(self.cells_per_dim), self.mi, marker='o', label='Data')
        plt.plot(smooth_x, cf.linear_fit(smooth_x, m, b), 'r-', label=f'Fit: $I(P;\\Phi) = - {m:.2f} \\log_2(\\epsilon) + {b:.2f}$')
        plt.axvspan(np.log2(self.cells_per_dim[self.start_fit_index]), np.log2(self.cells_per_dim[self.end_fit_index]), color='green', alpha=0.2, label='Fitting Region')
        plt.xlabel('$-\\log_2(\\epsilon)$')
        plt.ylabel('Mutual Information $I(P;\\Phi)$ [bits]')
        plt.title(f'Mutual Information $I(P;\\Phi) - {self.ensemble_name} Ensemble$')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(self.results_dir, f'mutual_information_fit_{self.ensemble_name}.png'))
        plt.close()

    def calc_density_matrix(self):
        rho_00 = np.average(1 - self.p)
        rho_01 = np.average(np.sqrt(self.p * (1 - self.p)) * np.exp(-1j * self.phi))
        rho_10 = np.average(np.sqrt(self.p * (1 - self.p)) * np.exp(1j * self.phi))
        rho_11 = np.average(self.p)

        return np.array([[rho_00, rho_01], 
                         [rho_10, rho_11]])
    
    def relative_entropy_of_coherence(self):
        rho = self.calc_density_matrix()
        rho_diag = np.diag(np.diag(rho))
        return cf.von_neumann_entropy(rho_diag) - cf.von_neumann_entropy(rho)
    
    def save_ent(self):
        headers = [f'$-\\log(\\epsilon)$', f'$H(P)$', f'$H(\\Phi)$', f'$H(P,\\Phi)$']

        ent_df = pd.DataFrame({
            headers[0]: np.log2(self.cells_per_dim),
            headers[1]: self.p_ent,
            headers[2]: self.phi_ent,
            headers[3]: self.joint_ent
        })

        ent_df.to_csv(os.path.join(self.results_dir, f'entropies_{self.ensemble_name}.csv'), index=False)

    def save_mi(self):
        headers = [f'$-\\log(\\epsilon)$', f'$I(P;\\Phi)$']

        mi_df = pd.DataFrame({
            headers[0]: np.log2(self.cells_per_dim),
            headers[1]: self.mi
        })

        mi_df.to_csv(os.path.join(self.results_dir, f'mutual_information_{self.ensemble_name}.csv'), index=False)
