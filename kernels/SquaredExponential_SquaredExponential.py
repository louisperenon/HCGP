from sympy import *
import numpy as np


class Convolution:
    def __init__(self, sp_hyps_SE1, sp_hyps_SE2):

        self.name = "SquaredExponential_SquaredExponential"
        self.n_dim = 4

        ### input vectors
        self.sp_X = Symbol("X", real=True)
        self.sp_Y = Symbol("Y", real=True)

        self.sp_sigma_1, self.sp_xi_1 = sp_hyps_SE1
        self.sp_sigma_2, self.sp_xi_2 = sp_hyps_SE2

        self.sp_hyps = np.array(
            [
                [self.sp_sigma_1, self.sp_xi_1],
                [self.sp_sigma_2, self.sp_xi_2],
            ]
        )

        ### Basis function convoluted
        self.sp_K = self.sp_sigma_1 * self.sp_sigma_2
        self.sp_K *= sqrt(
            (2 * self.sp_xi_1 * self.sp_xi_2) / (self.sp_xi_1 ** 2 + self.sp_xi_2 ** 2)
        )
        self.sp_K *= exp(
            -((Abs(self.sp_X - self.sp_Y)) ** 2)
            / (self.sp_xi_1 ** 2 + self.sp_xi_2 ** 2)
        )
