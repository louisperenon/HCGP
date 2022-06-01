from sympy import *
import numpy as np


class RationalQuadratic:
    def __init__(self, sigma="sigma", xi="xi", alpha="alpha"):

        self.name = "RationalQuadratic"
        self.n_dim = 3

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

        if isinstance(alpha, (int, float)):
            self.sp_alpha = alpha
        else:
            self.sp_alpha = Symbol(alpha, real=True)
            self.sp_hyps.append(self.sp_alpha)

        ### Kernel
        self.sp_K = self.sp_sigma ** 2 / (
            (1 + Abs(self.sp_X - self.sp_Y) ** 2 / (2 * self.sp_alpha)) ** self.sp_alpha
        )
