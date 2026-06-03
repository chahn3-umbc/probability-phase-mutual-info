# Probability-phase Mutual Information
---
This repo contains all of the relevant scripts and numerical data files that were used for the creation of figures and analysis of examples in the paper **Probability-phase mutual information** published in **Physical Review A**.

The structure of this repo is explained within this document in order to make navigating the various scripts and resources stored here simple.

## Canonical Ensemble Analysis (`canonical_ensemble_analysis/`)

This folder contains the scripts for computing and plotting the probability-phase mutual information (PPMI) for the canonical ensemble example (see **Figure 1**). The canonical ensemble is defined as:

$$q(p,\phi) = \frac{1}{Z_{\beta,g}}e^{-\beta h(p,\phi)}$$

where $h(p,\phi) = 1 - 2p + 2g\sqrt{p(1-p)}\cos\phi$ is one of the simplest example models which exhibits non-trivial PPMI.

### Files

- **`canonical_ensemble_numerics.py`**: Defines the `CanonicalEnsemble` class, which computes the PPMI for a given inverse temperature $\beta$ and coupling strength $g$ via direct numerical double integration (`scipy.integrate.dblquad`). Running the script as `__main__` sweeps a grid of $(\beta, g)$ values ($\beta \in [0.01, 100]$, $g \in [0, 2]$), computes the PPMI at each point, and saves the resulting matrix to `mutual_information_v_beta_g.csv`.

- **`canonical_ensemble_plotting.py`**: Reads `mutual_information_v_beta_g.csv` and renders the PPMI heatmap as a function of $\beta$ and $g$, saving the figure to `mutual_information_v_beta_g.png` (Figure 1 in the paper).

- **`mutual_information_v_beta_g.csv`**: Pre-computed PPMI values over the $(\beta, g)$ parameter grid, produced by `canonical_ensemble_numerics.py`.

- **`mutual_information_v_beta_g.png`**: The rendered heatmap figure produced by `canonical_ensemble_plotting.py`.

---

## DFS Gaussian Analysis (`dfs_gaussian_analysis/`)

This folder contains scripts analyzing the Fubini-Study (FS) Gaussian distribution — a Gaussian probability measure over qubit phase space $(p, \phi)$ defined with respect to the Fubini-Study metric. The PPMI of this distribution is studied as a function of the width parameter $\sigma$.

### Files

- **`dfs_gaussian_class.py`**: Defines the `DfsGaussianMeasure` class, which represents a Gaussian centered at $(p_0, \phi_0)$ with width $\sigma$ on the Bloch sphere. The class computes the joint density $q(p,\phi)$, the marginal densities $q_P(p)$ and $q_\Phi(\phi)$, and the PPMI via continuous numerical integration using `scipy.integrate.dblquad` and `scipy.integrate.quad`.

- **`df_gaussian_analysis.py`**: Main script for the continuous analysis. Iterates over a range of $\sigma$ values, computes the PPMI at each using `DfsGaussianMeasure`, and saves the plot of PPMI vs $\sigma$ to `dfsgaussian_mi_v_sigma.pdf`. Also generates and saves a heatmap of the joint distribution $q(p,\phi)$ to `dfsgaussian_joint_dist.pdf`. Run from within the `dfs_gaussian_analysis/` directory.

- **`dfs_gaussian_analysis-discrete.py`**: An alternative, vectorized, discrete-grid implementation of the same analysis. Uses a numpy meshgrid over $(p,\phi)$ with the correct Fubini-Study volume element ($dV_{FS} = \frac{1}{2}\,dp\,d\phi$) to compute the joint and marginal entropies numerically. Generates `dfsgaussian_mi_v_sigma_discrete.pdf` and `dfsgaussian_joint_dist.pdf` as cross-checks against the continuous results.

- **`df_gaussian_discrete_sampling.py`**: Generates discretized joint distribution samples $(p, \phi, q)$ for a given $(p_0, \phi_0, \sigma)$ on a regular grid and saves them to a CSV file. Used to produce the data files in `discrete_sampling_data/` for each $\sigma$ value, which are then fed into `EnsembleAnalysis` for the resolution-scaling PPMI computation.

- **`dfs_gaussian_joint_distribution.csv`**: Pre-computed joint distribution samples for a single set of parameters.

- **`discrete_sampling_data/dfs_gauss_sigma=*.csv`**: Discretized joint distribution data $(p, \phi, q)$ for $\sigma \in \{0.10, 0.26, 0.43, 0.59, 0.75, 0.92, 1.08, 1.24, 1.41, 1.57\}$, produced by `df_gaussian_discrete_sampling.py`. These files serve as input to the `EnsembleAnalysis` class for the resolution-scaling computation of PPMI.

---

## Ensemble Scaling Analysis (`ensemble_scaling_analysis/`)

This folder provides the shared infrastructure for computing PPMI via the resolution-scaling method. This approach estimates PPMI by histogramming a phase-space distribution $(p, \phi, q)$ at increasing grid resolutions $\epsilon$ and fitting the resulting MI-vs-resolution curve to extract the geometric mutual information (GMI), which is the $\epsilon \to 0$ extrapolation of the PPMI.

### Files

- **`ensemble_analysis_class.py`**: Defines the `EnsembleAnalysis` class. It loads pre-computed $(p, \phi, q)$ data from a CSV file, bins the distribution into 2D histograms at each resolution in `cells_per_dim` (a geometric sequence of bin counts per axis), computes the joint entropy $H(P,\Phi)$ and marginal entropies $H(P)$ and $H(\Phi)$ at each resolution, and fits the MI-vs-$\log_2(\epsilon^{-1})$ curve to a linear model $I(P;\Phi) \approx \mathfrak{D}_I \log_2(\epsilon^{-1}) + I_G(P;\Phi)$. The y-intercept $I_G(P;\Phi)$ is the GMI and the slope $\mathfrak{D}_I$ is the information dimension. Saves entropy and MI data to CSV files and generates fit plots. This class is used as the computational backbone for both the spiral ensemble and DFS Gaussian discrete analyses.

- **`comp_functions.py`**: Utility functions used by `EnsembleAnalysis`: coordinate conversions between $(p, \phi)$ and Bloch sphere or Cartesian coordinates, Shannon entropy calculation, von Neumann entropy calculation, and a linear fit function.

---

## Spiral Ensemble Analysis (`spiral_ensemble_analysis/`)

This folder contains the plotting script and pre-computed data for the spiral ensemble example, in which states are drawn from a spiral curve on the Bloch sphere with varying levels of angular noise $\delta$. The resolution-scaling PPMI data was generated using `EnsembleAnalysis`.

### Files

- **`spiral_ensemble_plotting.py`**: Reads the pre-computed MI scaling data from `spiral_ensemble_results/mutual_information_spiral_ensemble_delta=*.csv` for $\delta \in \{0.2, 0.4, 0.6, 0.8, 1.0\}$, fits each curve to a linear model over the fitting region defined by `start_indx` and `stop_indx`, and produces the combined multi-curve MI plot saved to `spiral_ensemble_results/spiral_ensemble_MI_plot.pdf`.

- **`spiral_ensemble_results/entropies_spiral_ensemble_delta=*.csv`**: Entropy data ($H(P)$, $H(\Phi)$, $H(P,\Phi)$) as a function of $\log_2(\epsilon^{-1})$ for each noise level $\delta$, generated by `EnsembleAnalysis`.

- **`spiral_ensemble_results/mutual_information_spiral_ensemble_delta=*.csv`**: PPMI-vs-$\log_2(\epsilon^{-1})$ data for each $\delta$, generated by `EnsembleAnalysis`.

- **`spiral_ensemble_results/mutual_information_fit_spiral_ensemble_delta=*.png`**: Individual MI fit plots for each $\delta$, generated by `EnsembleAnalysis`.

- **`spiral_ensemble_results/spiral_ensemble_MI_plot.pdf`**: The combined figure showing PPMI scaling curves for all $\delta$ values, as shown in the paper.

---

## Deep Thermalization Analysis (`deep_thermalization_analysis/`)

This folder contains the analysis for the deep thermalization example, where the projected ensemble of a many-body quantum system is used to compute the PPMI. The projected ensemble is constructed by measuring all qubits in subsystem B in the computational basis and collecting the resulting post-measurement states of a single target qubit A.

### Files

- **`gmi_analysis_for_ED.py`**: Core library for the deep thermalization analysis. Defines:
  - `get_projected_ensemble(psi, L, site_A)`: constructs the projected ensemble for a single qubit from a many-body state $|\psi\rangle$ of $L$ qubits by projecting onto all $z$-basis measurement outcomes of the remaining $L-1$ qubits.
  - `get_p_phi(ensemble)`: extracts the $(q, p, \phi)$ representation from each projected state.
  - `Ensemble` class: performs the full resolution-scaling PPMI computation (analogous to `EnsembleAnalysis`) and saves entropy and MI data to `ED_analysis_results/`.

- **`deep_thermalization_analysis.py`**: Main analysis script. Loads time-averaged projected ensemble data (pre-computed `.npz` files from exact diagonalization, stored in `projected_ensemble_data/`) for system sizes $N \in \{8, 10, 12, 14\}$, runs `Ensemble.analyze()` for each, and saves the resulting entropy and MI scaling data to `ED_analysis_results/`.

- **`scaling_analysis_plot_generator.py`**: Reads the saved MI data from `ED_analysis_results/Projected Ensemble N=*/mutual_info_data.csv` and generates the final comparison plot of PPMI vs. $-\log(\epsilon)$ across all system sizes, with shaded regions indicating fit quality zones.

- **`processed_data/mutual_info_data_N=*.csv`**: Pre-processed MI scaling data for $N \in \{8, 10, 12, 14\}$, produced by `deep_thermalization_analysis.py` and ready for plotting without re-running the full exact diagonalization pipeline.

- **`ED_analysis_results/Projected Ensemble N=*/`**: Per-system-size directories containing `joint_ent_data.csv`, `p_ent_data.csv`, `phi_ent_data.csv`, and `mutual_info_data.csv` — the full entropy and MI scaling results used to produce the figures in the paper.
