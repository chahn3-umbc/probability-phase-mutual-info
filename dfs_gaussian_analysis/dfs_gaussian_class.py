import numpy as np
from scipy import integrate

PHI_MIN = 0
PHI_MAX = 2 * np.pi
P_MIN = 0
P_MAX = 1


def dbl_integral(integrand):
    result, _ = integrate.dblquad(integrand, PHI_MIN, PHI_MAX, P_MIN, P_MAX)

    return result


def p_integral(integrand):
    result, _ = integrate.quad(integrand, P_MIN, P_MAX)

    return result


def phi_integral(integrand):
    result, _ = integrate.quad(integrand, PHI_MIN, PHI_MAX)

    return result


class DfsGaussianMeasure:
    def __init__(self, p0, phi0, sigma):
        self.p0 = p0
        self.phi0 = phi0
        self.sigma = sigma

        self.norm = self.compute_norm()

        self.joint_ent = None
        self.p_ent = None
        self.phi_ent = None
        self.mi = None

    def fubini_study_distance_squared(self, p, phi):
        arccos_arg = np.sqrt(
            (1 - p) * (1 - self.p0)
            + 2 * np.sqrt(p * self.p0 * (1 - p) * (1 - self.p0)) * np.cos(self.phi0 - phi)
            + p * self.p0
        )

        return (np.arccos(arccos_arg))**2

    def joint_integrand(self, p, phi):

        return np.exp(-self.fubini_study_distance_squared(p, phi)/(2 * self.sigma**2))

    def compute_norm(self):

        norm, _ = integrate.dblquad(self.joint_integrand, 0, 2 * np.pi, 0, 1)

        return norm

    def joint_measure(self, p, phi):

        return 1/self.norm * np.exp(-self.fubini_study_distance_squared(p, phi)/(2 * self.sigma**2))

    def p_measure(self, p):

        def integrand(phi):
            return self.joint_measure(p, phi)

        p_marg = phi_integral(integrand)

        return p_marg

    def phi_measure(self, phi):

        def integrand(p):
            return self.joint_measure(p, phi)

        phi_marg = p_integral(integrand)

        return phi_marg

    def compute_joint_entropy(self):

        def integrand(p, phi):
            return - self.joint_measure(p, phi) * np.log2(self.joint_measure(p, phi))

        self.joint_ent = dbl_integral(integrand)

        return self.joint_ent

    def compute_p_entropy(self):

        def integrand(p):
            return - self.p_measure(p) * np.log2(self.p_measure(p))

        self.p_ent = p_integral(integrand)

        return self.p_ent

    def compute_phi_entropy(self):

        def integrand(phi):
            return - self.phi_measure(phi) * np.log2(self.phi_measure(phi))

        self.phi_ent = phi_integral(integrand)

        return self.phi_ent

    def compute_mutual_information(self):

        self.mi = self.compute_p_entropy() + self.compute_phi_entropy() - self.compute_joint_entropy()

        return self.mi

