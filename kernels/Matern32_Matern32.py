from sympy import *
import numpy as np


class Convolution:
    def __init__(self, sp_hyps_Mat32_1, sp_hyps_Mat32_2):

        self.name = "Matern32_Matern32"
        self.n_dim = 4

        ### input vectors
        self.sp_X = Symbol("X", real=True)
        self.sp_Y = Symbol("Y", real=True)

        self.sp_sigma_1, self.sp_xi_1 = sp_hyps_Mat32_1
        self.sp_sigma_2, self.sp_xi_2 = sp_hyps_Mat32_2

        self.sp_hyps = np.array(
            [
                [self.sp_sigma_1, self.sp_xi_1],
                [self.sp_sigma_2, self.sp_xi_2],
            ]
        )

        ### Basis function convoluted
        ratio = Abs(self.sp_X - self.sp_Y)
        self.sp_K = self.sp_xi_1 * exp(-sqrt(3) * ratio / self.sp_xi_1)
        self.sp_K -= self.sp_xi_2 * exp(-sqrt(3) * ratio / self.sp_xi_2)
        self.sp_K *= 2*sqrt(self.sp_xi_1 * self.sp_xi_2) / (self.sp_xi_1**2 - self.sp_xi_2**2)
        self.sp_K *= self.sp_sigma_1 * self.sp_sigma_2
