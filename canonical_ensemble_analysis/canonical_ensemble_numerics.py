import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import integrate
import pandas as pd


class CanonicalEnsemble:

    def __init__(self, beta, g):
        self.beta = beta
        self.g = g
        self.Z_beta_g = self.compute_partition_function()

    def compute_partition_function(self):

        def integrand(p, phi):
            H = self.hamiltonian_expectation(p, phi)
            return np.exp(-self.beta * H)

        # Integrate over p ∈ [0,1] and φ ∈ [0, 2π]
        result, _ = integrate.dblquad(
            integrand, 0, 2 * np.pi, 0, 1
        )
        self.Z_beta_g = result
        return result

    def hamiltonian_expectation(self, p, phi):

        return 1 - 2 * p + 2 * self.g * np.sqrt(p * (1 - p)) * np.cos(phi)

    def joint_density(self, p, phi):

        H = self.hamiltonian_expectation(p, phi)
        return np.exp(-self.beta * H) / self.Z_beta_g

    def marginal_p_density(self, p):
        """Marginal density q_P,β,g(p) by integrating over φ"""

        def integrand(phi):
            return self.joint_density(p, phi)

        result, _ = integrate.quad(integrand, 0, 2 * np.pi)
        return result

    def marginal_phi_density(self, phi):
        """Marginal density q_Φ,β,g(φ) by integrating over p"""

        def integrand(p):
            return self.joint_density(p, phi)

        result, _ = integrate.quad(integrand, 0, 1)
        return result

    def joint_entropy(self):

        def joint_ent_func(p, phi):
            return -self.joint_density(p, phi) * np.log2(self.joint_density(p, phi))

        result, _ = integrate.dblquad(joint_ent_func, 0, 2*np.pi, 0, 1)

        return result

    def p_entropy(self):

        def p_ent_func(p):
            return -self.marginal_p_density(p) * np.log2(self.marginal_p_density(p))

        result, _ = integrate.quad(p_ent_func, 0, 1)

        return result

    def phi_entropy(self):

        def phi_ent_func(phi):
            return -self.marginal_phi_density(phi) * np.log2(self.marginal_phi_density(phi))

        result, _ = integrate.quad(phi_ent_func, 0, 2*np.pi)

        return result

    def compute_mutual_information(self):

        mi = self.p_entropy() + self.phi_entropy() - self.joint_entropy()

        return mi

    def plot_joint_density(self, n_points=100):

        p_vals = np.linspace(0.01, 0.99, n_points)
        phi_vals = np.linspace(0, 2 * np.pi, n_points)
        P_grid, Phi_grid = np.meshgrid(p_vals, phi_vals)

        density_grid = np.zeros_like(P_grid)
        for i in range(n_points):
            for j in range(n_points):
                density_grid[i, j] = self.joint_density(P_grid[i, j], Phi_grid[i, j])

        plt.figure(figsize=(10, 8))
        plt.contourf(P_grid, Phi_grid, density_grid, levels=20, cmap='viridis')
        plt.colorbar(label='Probability Density')
        plt.xlabel('Probability p')
        plt.ylabel('Phase φ')
        plt.title(f'Joint Density q(p,φ) for β={self.beta}, g={self.g}')
        plt.show()


def analyze_mutual_information_vs_parameters():
    """
    Analyze how mutual information varies with β and g parameters
    """
    # Parameter ranges
    beta_vals = np.linspace(0.01, 100, 51)  # 0.01 to 100
    g_vals = np.linspace(0, 2, 51)  # 0 to 2

    results = {}

    print("Computing mutual information for different parameter values...")
    print("This may take a few minutes...")

    for g in g_vals:
        mi_vals = []
        for beta in beta_vals:
            try:
                ensemble = CanonicalEnsemble(beta, g)
                mi = ensemble.compute_mutual_information()
                mi_vals.append(mi)
                print(f"β={beta:.2f}, g={g:.2f}: I(P;Φ) = {mi:.4f}")
            except Exception as e:
                print(f"Error for β={beta:.2f}, g={g:.2f}: {e}")
                mi_vals.append(0)

        results[g] = mi_vals

    # Plot results
    plt.figure(figsize=(12, 8))

    for g in g_vals[::2]:  # Plot every other g value for clarity
        if g in results:
            plt.plot(beta_vals, results[g], label=f'g={g:.1f}')

    plt.xlabel('Inverse Temperature β')
    plt.ylabel('Mutual Information I(P; Φ)')
    plt.title('Mutual Information vs Temperature for Different Coupling Strengths')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # Plot heatmap
    plt.figure(figsize=(12, 8))
    mi_matrix = np.array([results[g] for g in g_vals])

    plt.imshow(mi_matrix, origin='lower', cmap='rocket')
    plt.colorbar(label='Mutual Information I(P; Φ)')

    # Set tick labels
    beta_ticks = np.arange(0, len(beta_vals), 2)
    g_ticks = np.arange(0, len(g_vals), 2)

    # plt.xticks(beta_ticks, [f'{beta_vals[i]:.1f}' for i in beta_ticks])
    # plt.yticks(g_ticks, [f'{g_vals[i]:.1f}' for i in g_ticks])

    plt.xlabel('Inverse Temperature β')
    plt.ylabel('Coupling Strength g')
    plt.title('Mutual Information I(P; Φ) Heatmap')
    plt.savefig('mutual_info_v_beta_g_lower_res.pdf')
    plt.show()

    mi_df = pd.DataFrame(mi_matrix)
    mi_df.to_csv('mutual_information_v_beta_g.csv')


if __name__ == "__main__":
    # Example usage
    print("=== Canonical Ensemble Mutual Information Calculator ===\n")

    # Test case 1: g=0 (should give MI ≈ 0)
    print("Test Case 1: g=0 (no coupling)")
    ensemble_g0 = CanonicalEnsemble(beta=1.0, g=0.0)
    mi_g0 = ensemble_g0.compute_mutual_information()
    print(f"β=1.0, g=0.0: I(P;Φ) = {mi_g0:.6f} (should be ≈ 0)")

    # Test case 2: g≠0 (should give MI > 0)
    print("\nTest Case 2: g≠0 (with coupling)")
    ensemble_g1 = CanonicalEnsemble(beta=2.0, g=1.0)
    mi_g1 = ensemble_g1.compute_mutual_information()
    print(f"β=2.0, g=1.0: I(P;Φ) = {mi_g1:.6f}")

    # Plot joint density for visualization
    print("\nPlotting joint density...")
    ensemble_g1.plot_joint_density()

    # Comprehensive analysis
    response = input("\nRun comprehensive parameter analysis? (y/n): ")
    if response.lower() == 'y':
        analyze_mutual_information_vs_parameters()

    print("\nScript completed!")
