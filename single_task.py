import numpy as np
from sympy import *


class Covfunc:
    """Covariance function class"""

    def __init__(self, kernel, derivatives=False, gradient=False):

        # Name of the covariance function
        self.name = kernel.name

        # Input vectors
        self.sp_X = kernel.sp_X
        self.sp_Y = kernel.sp_Y

        # Hyper parameters
        self.sp_hyps = kernel.sp_hyps

        # Kernel
        self.sp_K = kernel.sp_K

        # Analytical computations
        self.set_K(self.sp_hyps)
        if derivatives:
            self.set_K_derivatives(self.sp_hyps)
        if gradient:
            self.set_K_gradient(self.sp_hyps)

    ### Overloaders
    def _merge(self, other):
        self.sp_hyps = np.append(self.sp_hyps, other.sympy_hyps)
        if self.derivatives:
            self.set_sympy_derivatives()
        self.sp_expressions["K"] = str(self.sp_K)

    def __add__(self, other):
        self.sp_K = Add(self.sp_K + other.sympy_K, evaluate=False)
        self._merge(other)
        return self

    def __mul__(self, other):
        self.sp_K = Mul(self.sp_K, other.sympy_K, evaluate=False)
        self._merge(other)
        return self

    def __iadd__(self, other):
        self.__add__(other)
        return self

    def __imul__(self, other):
        self.__mul__(other)
        return self

    ### Setters
    def set_K(self, sympy_hyps):
        self.sp_expressions = {"K": str(self.sp_K)}
        self.K = lambdify(
            (sympy_hyps, self.sp_X, self.sp_Y), self.sp_K, modules="numpy"
        )

    def set_K_derivatives(self, sympy_hyps):
        self.sp_dK_dY = simplify(self.sp_K.diff(self.sp_Y, real=True))
        self.sp_d2K_dY2 = simplify(self.sp_dK_dY.diff(self.sp_Y, real=True))
        self.sp_d2K_dXdY = simplify(self.sp_dK_dY.diff(self.sp_X, real=True))
        # self.sp_d4K_dX2dY2 = simplify(
        #     self.sp_d2K_dY2.diff(self.sp_X, real=True).diff(self.sp_X, real=True)
        # )
        print(self.sp_d2K_dXdY)

        self.sp_expressions["dK_dY"] = str(self.sp_dK_dY)
        self.sp_expressions["d2K_dY2"] = str(self.sp_d2K_dY2)
        self.sp_expressions["d2K_dXdY"] = str(self.sp_d2K_dXdY)
        # print(str(self.sp_d2K_dXdY))
        # self.sp_expressions["d4K_dX2dY2"] = str(self.sp_d4K_dX2dY2)

        self.dK_dY_ = lambdify(
            (sympy_hyps, self.sp_X, self.sp_Y), self.sp_dK_dY, modules="numpy"
        )
        self.d2K_dY2_ = lambdify(
            (sympy_hyps, self.sp_X, self.sp_Y),
            self.sp_d2K_dY2,
            modules="numpy",
        )
        self.d2K_dXdY_ = lambdify(
            (sympy_hyps, self.sp_X, self.sp_Y),
            self.sp_d2K_dXdY,
            # modules="numpy",
        )
        # self.d4K_dX2dY2_ = lambdify(
        #     (sympy_hyps, self.sp_X, self.sp_Y),
        #     self.sp_d4K_dX2dY2,
        #     modules="numpy",
        # )

    def set_K_grad(self, sympy_hyps):
        self.sp_grad_K = self.sp_K
        for hyp in sympy_hyps:
            self.sp_grad_K = simplify(self.sp_grad_K.diff(hyp))

        self.sp_expressions["grad_K"] = str(self.sp_grad_K)
        self.sp_grad_K_ = lambdify(
            (sympy_hyps, self.sp_X, self.sp_Y), self.sp_grad_K, modules="numpy"
        )

    ### Getters
    def _get_XY(self, X, Y):
        if isinstance(Y, float) or isinstance(X, float):
            X = X
        else:
            X = np.transpose(np.tile(X, (len(Y), 1)))
        return X, Y

    def get_K(self, p, X, Y):
        X, Y = self._get_XY(X, Y)
        res = self.K(p, X, Y)
        return res

    def get_dK_dY(self, p, X, Y):
        X, Y = self._get_XY(X, Y)
        res = -self.dK_dY_(p, X, Y)
        return res

    def get_d2K_dY2(self, p, X, Y):
        X, Y = self._get_XY(X, Y)
        res = self.d2K_dY2_(p, X, Y)
        return res

    def get_d2K_dXdY(self, p, X, Y):
        X, Y = self._get_XY(X, Y)
        res = self.d2K_dXdY_(p, X, Y)
        return res

    # def get_d4K_dX2dY2(self, p, X, Y):
    #     X, Y = self._get_XY(X, Y)
    #     res = self.d4K_dX2dY2_(p, X, Y)
    #     return res

    ### Printers
    def print_sympy_expressions(self):
        for key, value in self.sp_expressions.items():
            print("\n >> " + key + ": \n" + value)
        print("\n")


################################################################################
### Functions for engine.py
################################################################################
def get_sampling_info(gp_info):
    sampling_info = {
        "hyp_sampled": [],
        "inis": [],
        "ranges": np.empty((0, 2)),
        "sampling_invfuncs": [],
    }
    for i, par in enumerate(gp_info["hyperpars"]):
        # --
        sampling_info["hyp_sampled"].append(par)
        # --
        sampling_info["ranges"] = np.vstack(
            (
                sampling_info["ranges"],
                np.array(gp_info["hyperpars"][par]["range"]),
            )
        )
        # --
        if "ini" in gp_info["hyperpars"][par]:
            sampling_info["inis"].append(gp_info["hyperpars"][par]["ini"])
        else:
            sampling_info["inis"].append(
                (
                    gp_info["hyperpars"][par]["range"][1]
                    - gp_info["hyperpars"][par]["range"][0]
                )
                / 2.0
            )
        # --
        if "sampling_invfunc" in gp_info["hyperpars"][par]:
            sampling_info["sampling_invfuncs"].append(
                gp_info["hyperpars"][par]["sampling_invfunc"]
            )
        else:
            sampling_info["sampling_invfuncs"].append(lambda x: x)

    return sampling_info


def get_reconstruction(x, gp, gp_info):
    mean, cov = gp.predict(x, predict_cov=True)
    err = np.sqrt(np.diag(cov))
    res = {
        "x": x,
        "mean": mean,
        "err": err,
        "cov": cov,
    }
    if gp_info["derivatives"]:
        mean, cov = gp.predict_d1(x, predict_cov=True)
        err = np.sqrt(np.diag(cov))
        cov_01 = gp.predict_cov_01(x)
        res.update(
            {
                "mean_d1": mean,
                "err_d1": err,
                "cov_d1": cov,
                "cov_01": cov_01,
            }
        )
        # mean, cov = gp.predict_d2(x, predict_cov=True)
        # err = np.sqrt(np.diag(cov))
        # res.update(
        #     {
        #         "mean_d2": mean,
        #         "err_d2": err,
        #         "cov_d2": cov,
        #     }
        # )
    return res


################################################################################
### Functions for master.py
################################################################################
def get_covfunc(gp_info):
    return Covfunc(
        gp_info["kernel"],
        derivatives=gp_info["derivatives"],
        gradient=gp_info["gradient"],
    )


def get_mean_priors(gp_info, data):
    """Mean priors on the reconstruction getter"""

    null_func = lambda x: np.zeros(len(x))
    priors = {
        "mean": null_func,
        "mean_d1": null_func,
        "mean_d2": null_func,
    }

    names = ["mean", "mean_d1", "mean_d2"]
    if "mean_priors" in gp_info:
        for i, name in enumerate(names):
            if name in gp_info["mean_priors"]:
                priors[i] = gp_info["mean_priors"][name]

    priors["Y"] = data["y"] - priors["mean"](data["x"])
    priors["YT"] = np.transpose(priors["Y"])

    return priors


def get_n_data(data):
    """Returns the number of data"""
    return len(data["x"])
