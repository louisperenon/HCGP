import numpy as np

# from .defaults import *


def mat_inv(mat, method="QR"):
    """Choice of the matrix inversion method"""
    if method == "QR":
        Q, R = np.linalg.qr(mat)
        return np.dot(Q, np.linalg.inv(R.T))


class GP:
    """This Gaussian Process regression code has been inspired by:
    - The Bible:
        Gaussian Processes for Machine Learning
        C. E. Rasmussen & C. K. I. Williams,
        Gaussian Processes for Machine Learning, the MIT Press, 2006,
        ISBN 026218253X. c 2006 Massachusetts Institute of Technology.
        www.GaussianProcess.org/gpml

    - GaPP code: arXiv:1204.2832

    - arXiv:1912.04325

    Many thanks to all.
    """

    def __init__(self, gp_info, data):

        # > Set data
        self.data = data
        self.n_data = gp_info["mode"].get_n_data(data)

        # > Set mean priors
        self.priors = gp_info["mode"].get_mean_priors(gp_info, data)

        # > Set covariance function
        self.covfunc = gp_info["mode"].get_covfunc(gp_info)
        test = np.all([isinstance(hyp, (int, float)) for hyp in self.covfunc.sp_hyps])
        if test:  # needed if hyps given as numbers in the covfunc instance
            self.check_hyp_values([])

        # > Setting the matrix inversion method
        if "matrix_inversion_method" in gp_info:
            self.mat_inv_method = gp_info["matrix_inversion_method"]
        else:
            self.mat_inv_method = "QR"

        # Setting default attributes
        self.hyp_values = None

    def check_hyp_values(self, p):
        """Checker that the K+C matrix is invertible"""
        self.cov_kc = (
            self.covfunc.get_K(p, self.data["x"], self.data["x"]) + self.data["cov"]
        )
        self.invcov_kc = mat_inv(self.cov_kc, self.mat_inv_method)
        self.invcov_kc_test = np.max(
            np.abs(np.dot(self.cov_kc, self.invcov_kc) - np.eye(self.n_data))
        )
        if self.invcov_kc_test > 1e-8:
            raise ValueError(
                "Inversion of K+C not precise enough: \
                max(abs(KC*KC^-Id)) > 1e-8"
            )

    def set_hyp_values(self, p):
        """Hyperparameter values setter"""
        self.hyp_values = p.values()
        self.check_hyp_values(self.hyp_values)

    def get_hyp_values(self):
        """Hyperparameter values getter"""
        return self.hyp_values

    def log_marginal_likelihood(self):
        """Computes log marginal likelihood"""
        res = np.linalg.multi_dot([self.priors["YT"], self.invcov_kc, self.priors["Y"]])
        res += np.linalg.slogdet(self.cov_kc)[1]
        res += self.n_data * np.log(2 * np.pi)
        res *= -0.5
        return res

    def predict(self, pred_x, predict_cov=True):
        """Computes the mean and covariance of the latent function"""
        cov_xstar_x = self.covfunc.get_K(self.hyp_values, pred_x, self.data["x"])

        mean = self.priors["mean"](pred_x) + np.linalg.multi_dot(
            [cov_xstar_x, self.invcov_kc, self.priors["Y"]]
        )
        if predict_cov:
            cov = self.covfunc.get_K(
                self.hyp_values, pred_x, pred_x
            ) - np.linalg.multi_dot(
                [cov_xstar_x, self.invcov_kc, np.transpose(cov_xstar_x)]
            )
            return mean, cov
        else:
            return mean

    def predict_d1(self, pred_x, predict_cov=True):
        """Computes the mean and covariance of the first derivative of the latent function"""
        dk_dy = self.covfunc.get_dK_dY(self.hyp_values, pred_x, self.data["x"])
        mean = self.priors["mean_d1"](pred_x) + np.linalg.multi_dot(
            [dk_dy, self.invcov_kc, self.priors["Y"]]
        )
        mean = np.linalg.multi_dot([dk_dy, self.invcov_kc, self.priors["Y"]])

        if predict_cov:
            cov = self.covfunc.get_d2K_dXdY(
                self.hyp_values, pred_x, pred_x
            ) - np.linalg.multi_dot(
                [
                    dk_dy,
                    self.invcov_kc,
                    np.transpose(dk_dy),
                ]
            )
            return mean, cov
        else:
            return mean

    def predict_d2(self, pred_x, predict_cov=True):
        """Computes the mean and covariance of the second derivative of the latent function"""
        d2k_dy2 = self.covfunc.get_d2K_dY2(self.hyp_values, pred_x, self.data["x"])
        mean = self.priors["mean_d2"](pred_x) + np.linalg.multi_dot(
            [d2k_dy2, self.invcov_kc, self.priors["Y"]]
        )
        if predict_cov:
            cov = self.covfunc.get_d4K_dX2dY2(
                self.hyp_values, pred_x, pred_x
            ) - np.linalg.multi_dot(
                [
                    d2k_dy2,
                    self.invcov_kc,
                    np.transpose(d2k_dy2),
                ]
            )
            return mean, cov
        else:
            return mean

    def predict_cov_01(self, pred_x):
        """Computes the covariance matrix between latent function and its first derivative"""
        cov_xstar_x = self.covfunc.get_K(self.hyp_values, pred_x, self.data["x"])
        d_cov_xstar_xstar = self.covfunc.get_dK_dY(self.hyp_values, pred_x, pred_x)
        d_cov_x_xstar = self.covfunc.get_dK_dY(self.hyp_values, self.data["x"], pred_x)
        cov_01 = d_cov_xstar_xstar - np.dot(
            np.dot(cov_xstar_x, self.invcov_kc), d_cov_x_xstar
        )
        return cov_01
