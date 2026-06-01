import numpy as np


def to_spherical(p_vals, phi_vals):
    theta_vals = 2 * np.arccos(np.sqrt(1 - p_vals))
    return theta_vals, phi_vals


def to_cartesian(p_vals, phi_vals):
    theta_vals, phi_vals = to_spherical(p_vals, phi_vals)
    x = np.sin(theta_vals) * np.cos(phi_vals)
    y = np.sin(theta_vals) * np.sin(phi_vals)
    z = np.cos(theta_vals)
    return x, y, z


def calc_entropy(prob_dist):
    # Calculate the entropy of a probability distribution
    prob_dist = prob_dist[prob_dist > 0]  # Remove zero probabilities
    return -np.sum(prob_dist * np.log2(prob_dist))


def von_neumann_entropy(rho):
    # Calculate the von Neumann entropy of a quantum state
    eigvals = np.linalg.eigvalsh(rho)
    eigvals = eigvals[eigvals > 0]  # Remove zero eigenvalues
    return -np.sum(eigvals * np.log2(eigvals))


def linear_fit(x, m, b):
    return m * x + b
