# Numerical computation of joint entropy, marginal entropies, and mutual information
# for the Gaussian on the Fubini-Study distance in (p, phi) coordinates.
#
# Volume element used: dV_FS = (1/2) dp dphi (as you corrected).
# The code computes:
#   - H(p,phi) = -∫ q log q dV_FS
#   - H(p)   = -∫ q_p log q_p dp   (q_p is marginal w.r.t. dp)
#   - H(phi) = -∫ q_phi log q_phi dphi (q_phi marginal w.r.t. dphi)
#   - I(p;phi) = H(p) + H(phi) - H(p,phi)
#
# Notes:
#  - q(p,phi) = (1/Z) * exp(-D_FS^2/(2 sigma^2)), normalized w.r.t. dV_FS.
#  - Results are reported in nats and bits.
#  - Grid resolution (Np, Nphi) can be increased for higher accuracy.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------
# User-adjustable params
# ---------------------
sigma = 0.3         # kernel width (in radians, since D_FS is an angle)
p0 = 0.5            # reference p' (center of the Gaussian)
phi0 = 0.0          # reference phi'
Np = 200            # number of p grid points
Nphi = 200          # number of phi grid points
# ---------------------

# Grid setup
p_vals = np.linspace(0.0, 1.0, Np)
phi_vals = np.linspace(0.0, 2*np.pi, Nphi, endpoint=False)
dp = p_vals[1] - p_vals[0]
dphi = 2*np.pi / Nphi

P = p_vals[:, None]        # shape (Np,1)
Phi = phi_vals[None, :]    # shape (1,Nphi)

# Fubini-Study distance (vectorized)
# inner product <psi(P,Phi) | psi(p0,phi0)> = sqrt(P*p0) + sqrt((1-P)*(1-p0)) * exp(i*(phi0 - Phi))
overlap = np.sqrt(P * p0) + np.sqrt((1.0 - P) * (1.0 - p0)) * np.exp(1j * (phi0 - Phi))
mag = np.abs(overlap)
mag = np.clip(mag, 0.0, 1.0)   # numerical safety
D = np.arccos(mag)             # Fubini-Study distance (0..pi/2 for qubits)

# unnormalized density (function on coordinates)
tilde = np.exp(-D**2 / (2.0 * sigma**2))

# Fubini-Study volume element in p,phi coordinates:
measure = 0.5   # dV_FS = (1/2) dp dphi

# Normalize with respect to dV_FS
Z = np.sum(tilde) * measure * dp * dphi
q = tilde / Z   # q is density w.r.t. dV_FS: ∫ q dV_FS = 1

# Sanity check: total probability
total_prob = np.sum(q) * measure * dp * dphi

# Joint entropy H(p,phi) = -∫ q log q dV_FS
eps = 1e-20
H_joint = -np.sum(q * np.log(q + eps)) * measure * dp * dphi

# Marginals with respect to the coordinate measures dp and dphi:
# q_p(p)   = ∫ q(p,phi) dV_FS / dp  = measure * ∫ q dphi
# q_phi(phi) = ∫ q(p,phi) dV_FS / dphi = measure * ∫ q dp
q_p = measure * np.sum(q, axis=1) * dphi    # shape (Np,), integrates over dp to 1
q_phi = measure * np.sum(q, axis=0) * dp    # shape (Nphi,), integrates over dphi to 1

# small renormalization to enforce normalization within numerical precision
q_p /= np.sum(q_p) * dp
q_phi /= np.sum(q_phi) * dphi

H_p = -np.sum(q_p * np.log(q_p + eps)) * dp
H_phi = -np.sum(q_phi * np.log(q_phi + eps)) * dphi

I = H_p + H_phi - H_joint

# convert to bits
ln2 = np.log(2.0)
results = pd.DataFrame({
    'nats': [H_joint, H_p, H_phi, I],
    'bits': [H_joint/ln2, H_p/ln2, H_phi/ln2, I/ln2]
}, index=['H(p,phi)','H(p)','H(phi)','I(p;phi)'])

print("Parameters:")
print(f"  sigma = {sigma}, reference (p',phi') = ({p0:.3f}, {phi0:.3f})")
print(f"  grid: Np={Np}, Nphi={Nphi}, dp={dp:.4e}, dphi={dphi:.4e}")
print()
print(f"Normalization check (should be ~1): {total_prob:.12f}")
print()
print("Entropy / mutual information:")
print(results.to_string(float_format='{:0.6f}'.format))

# expose some arrays for further inspection if desired
# (q, p_vals, phi_vals, q_p, q_phi) are available in the notebook namespace

# 1. Heatmap of joint distribution q(p,phi)
plt.figure(figsize=(7,5))
plt.imshow(q, extent=[0, 2*np.pi, 0, 1],
           origin='lower', aspect='auto',
           cmap='viridis')
plt.colorbar(label='q(p,φ)')
plt.xlabel("φ")
plt.ylabel("p")
plt.title("Joint distribution q(p,φ)")
plt.savefig('dfs_gaussian_analysis/dfsgaussian_joint_dist.pdf')
plt.show()

# 2. Marginals
plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
plt.plot(p_vals, q_p, label="q(p)")
plt.xlabel("p")
plt.ylabel("density")
plt.title("Marginal over p")
plt.legend()

plt.subplot(1,2,2)
plt.plot(phi_vals, q_phi, label="q(φ)")
plt.xlabel("φ")
plt.ylabel("density")
plt.title("Marginal over φ")
plt.legend()

plt.tight_layout()
plt.show()

# 3. Mutual Information vs sigma
def compute_MI(sigma, Np=200, Nphi=200):
    p_vals = np.linspace(0.0, 1.0, Np)
    phi_vals = np.linspace(0.0, 2*np.pi, Nphi, endpoint=False)
    dp = p_vals[1] - p_vals[0]
    dphi = 2*np.pi / Nphi
    P = p_vals[:, None]
    Phi = phi_vals[None, :]
    overlap = np.sqrt(P * p0) + np.sqrt((1-P)*(1-p0)) * np.exp(1j*(phi0 - Phi))
    mag = np.abs(overlap)
    D = np.arccos(np.clip(mag,0,1))
    tilde = np.exp(-D**2/(2*sigma**2))
    measure = 0.5
    Z = np.sum(tilde) * dp * dphi
    q = tilde / Z
    H_joint = -np.sum(q * np.log(q+1e-20)) * dp * dphi
    q_p = np.sum(q, axis=1) * dphi
    q_phi = np.sum(q, axis=0) * dp
    q_p /= np.sum(q_p) * dp
    q_phi /= np.sum(q_phi) * dphi
    H_p = -np.sum(q_p * np.log(q_p+1e-20)) * dp
    H_phi = -np.sum(q_phi * np.log(q_phi+1e-20)) * dphi
    return H_p + H_phi - H_joint

sigma_vals = np.linspace(0.05, np.pi/2, 20)
I_vals = [compute_MI(s) for s in sigma_vals]

plt.figure(figsize=(6,4))
plt.plot(sigma_vals, np.array(I_vals)/np.log(2), marker='o')
plt.xlabel("σ")
plt.ylabel("I(p;φ) [bits]")
plt.title("Mutual Information vs σ")
plt.grid(True)
plt.savefig('dfs_gaussian_analysis/dfsgaussian_mi_v_sigma_discrete.pdf')
plt.show()

