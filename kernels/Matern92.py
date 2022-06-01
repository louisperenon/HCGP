from sympy import *
import numpy as np


class Matern92:
    def __init__(self, sigma="sigma", xi="xi"):

        self.name = "Matern92"
        self.n_dim = 2

        ### input vectors
        self.sp_X = Symbol("X", real=True)
        self.sp_Y = Symbol("Y", real=True)

        ### Hyperparameters
        self.sp_hyps = []
        if isinstance(sigma, (int, float)):
            self.sp_sigma = sigma
        else:
            self.sp_sigma = Symbol(sigma, real=True)
            self.sp_hyps.append(self.sp_sigma)

        if isinstance(xi, (int, float)):
            self.sp_xi = xi
        else:
            self.sp_xi = Symbol(xi, real=True)
            self.sp_hyps.append(self.sp_xi)

        ### Kernel
        ratio = Abs(self.sp_X - self.sp_Y) / self.sp_xi
        self.sp_K = (
            self.sp_sigma ** 2
            * (
                1
                + 3 * ratio
                + 27.0 / 7 * ratio ** 2
                + 18.0 / 7 * ratio ** 3
                + 27.0 / 35.0 * ratio ** 4
            )
            * exp(-3 * ratio)
        )
        # self.sp_K = self.sp_sigma ** 2 * (1 + 3 * dist + 27./7 * dist ** 2 + 18./7 * dist ** 3 + 27./35. * dist ** 4   ) * exp(-3 * dist)
