from sympy import *
import numpy as np


class Matern52:
    def __init__(self, sigma="sigma", xi="xi"):

        self.name = "Matern52"
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
            * (1 + sqrt(5) * ratio + 5.0 / 3.0 * ratio ** 2)
            * exp(-sqrt(5) * ratio)
        )
