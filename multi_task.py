import numpy as np
from scipy.linalg import block_diag
from sympy import *

from .kernels.__init__ import get_convolution
from .single_task import get_covfunc as stgp_get_covfunc


# *******************************************************************
# Mtgp covfunc class
# https://www.ijcai.org/Proceedings/11/Papers/238.pdf
# *******************************************************************
class Covfunc:
    """Covariance function class for single task GPs"""

    def __init__(self, gp_info):

        ### Number of function to reconstruct simultaneously
        self.n_tasks = gp_info["n_tasks"]

        ### input vectors
        self.sp_X = Symbol("X", real=True)
        self.sp_Y = Symbol("Y", real=True)

        # Analytical computations
        self.set_K(gp_info)

    ###
    def set_K(self, gp_info):

        self.covfuncs = np.empty([self.n_tasks, self.n_tasks]).tolist()
        self.sp_K = np.empty([self.n_tasks, self.n_tasks]).tolist()
        self.K = np.empty([self.n_tasks, self.n_tasks]).tolist()
        self.sp_hyps = []
        self.n_dims = []

        # Setting the diagonal terms to the single task kernel
        for i in range(self.n_tasks):
            self.sp_hyps.append(gp_info["func_" + str(i + 1)]["kernel"].sp_hyps)

            if not "derivatives" in gp_info["func_" + str(i + 1)]:
                gp_info["func_" + str(i + 1)]["derivatives"] = False
            if not "gradient" in gp_info["func_" + str(i + 1)]:
                gp_info["func_" + str(i + 1)]["gradient"] = False

            self.covfuncs[i][i] = stgp_get_covfunc(gp_info["func_" + str(i + 1)])
            self.sp_K[i][i] = self.covfuncs[i][i].sp_K
            self.n_dims.append(gp_info["func_" + str(i + 1)]["kernel"].n_dim)

        self.cumul_dims = flatten([np.cumsum(self.n_dims)])

        for i in range(self.n_tasks):
            for j in range(self.n_tasks):
                if i < j:
                    self.sp_K[i][j] = get_convolution(
                        self.covfuncs[i][i], self.covfuncs[j][j]
                    ).sp_K

        if "derivatives" in gp_info["func_" + str(i + 1)]:
            self.set_K_derivatives()
        if "gradient" in gp_info["func_" + str(i + 1)]:
            self.set_K_gradient()

        # Lambdifying the multi task kernel
        for i in range(self.n_tasks):
            for j in range(self.n_tasks):
                if i == j:
                    res = self.covfuncs[i][j]
                    res.set_K(self.sp_hyps[i])
                    self.K[i][j] = res

                if i < j:
                    self.K[i][j] = lambdify(
                        (
                            flatten([self.sp_hyps[i], self.sp_hyps[j]]),
                            self.sp_X,
                            self.sp_Y,
                        ),
                        self.sp_K[i][j],
                        modules="numpy",
                    )

    def set_K_derivatives(self):
        self.sp_dK_dY = np.empty([self.n_tasks, self.n_tasks]).tolist()
        self.sp_d2K_dY2 = np.empty([self.n_tasks, self.n_tasks]).tolist()
        self.sp_d2K_dXdY = np.empty([self.n_tasks, self.n_tasks]).tolist()
        # self.sp_d4K_dX2dY2 = np.empty([self.n_tasks, self.n_tasks]).tolist()
        for i in range(self.n_tasks):
            for j in range(self.n_tasks):
                if i <= j:
                    self.sp_dK_dY[i][j] = simplify(self.sp_K[i][j].diff(self.sp_Y))
                    self.sp_d2K_dY2[i][j] = simplify(
                        self.sp_dK_dY[i][j].diff(self.sp_Y)
                    )
                    self.sp_d2K_dXdY[i][j] = simplify(
                        self.sp_dK_dY[i][j].diff(self.sp_X)
                    )
                    # self.sp_d4K_dX2dY2[i][j] = simplify(
                    #     self.sp_d2K_dY2[i][j].diff(self.sp_X).diff(self.sp_X)
                    # )

        self.dK_dY_ = np.empty([self.n_tasks, self.n_tasks]).tolist()
        self.d2K_dY2_ = np.empty([self.n_tasks, self.n_tasks]).tolist()
        self.d2K_dXdY_ = np.empty([self.n_tasks, self.n_tasks]).tolist()
        # self.d4K_dX2dY2_ = np.empty([self.n_tasks, self.n_tasks]).tolist()

        for i in range(self.n_tasks):
            for j in range(self.n_tasks):
                if i == j:
                    self.dK_dY_[i][j] = lambdify(
                        (self.sp_hyps[i], self.sp_X, self.sp_Y),
                        self.sp_dK_dY[i][j],
                        modules="numpy",
                    )
                    self.d2K_dY2_[i][j] = lambdify(
                        (self.sp_hyps[i], self.sp_X, self.sp_Y),
                        self.sp_d2K_dY2[i][j],
                        modules="numpy",
                    )
                    self.d2K_dXdY_[i][j] = lambdify(
                        (self.sp_hyps[i], self.sp_X, self.sp_Y),
                        self.sp_d2K_dXdY[i][j],
                        modules="numpy",
                    )
                    # self.d4K_dX2dY2_[i][j] = lambdify(
                    #     (self.sp_hyps[i], self.sp_X, self.sp_Y),
                    #     self.sp_d4K_dX2dY2[i][j],
                    #     modules="numpy",
                    # )

                if i < j:
                    self.dK_dY_[i][j] = lambdify(
                        (
                            flatten([self.sp_hyps[i], self.sp_hyps[j]]),
                            self.sp_X,
                            self.sp_Y,
                        ),
                        self.sp_dK_dY[i][j],
                        modules="numpy",
                    )
                    self.d2K_dY2_[i][j] = lambdify(
                        (
                            flatten([self.sp_hyps[i], self.sp_hyps[j]]),
                            self.sp_X,
                            self.sp_Y,
                        ),
                        self.sp_d2K_dY2[i][j],
                        modules="numpy",
                    )
                    self.d2K_dXdY_[i][j] = lambdify(
                        (
                            flatten([self.sp_hyps[i], self.sp_hyps[j]]),
                            self.sp_X,
                            self.sp_Y,
                        ),
                        self.sp_d2K_dXdY[i][j],
                        modules="numpy",
                    )
                    # self.d4K_dX2dY2_[i][j] = lambdify(
                    #     (
                    #         flatten([self.sp_hyps[i], self.sp_hyps[j]]),
                    #         self.sp_X,
                    #         self.sp_Y,
                    #     ),
                    #     self.sp_d4K_dX2dY2[i][j],
                    #     modules="numpy",
                    # )

    def set_K_gradient(self):
        pass
        # self.sp_grad_K = self.sp_K
        # for i in range(self.n_tasks):
        #     for j in range(self.n_tasks):
        #         if i <= j:
        #             self.sp_grad_K[i][j] = simplify(self.sp_K[i][j].diff(self.sp_Y))
        #
        # self.sp_grad_K_ = lambdify(
        #     (sympy_hyps, self.sp_X, self.sp_Y), self.sp_grad_K, modules="numpy"
        # )

    ### Getters
    def _get_XY(self, X, Y):
        if isinstance(Y, float) or isinstance(X, float):
            X = X
        else:
            X = np.transpose(np.tile(X, (len(Y), 1)))
        return X, Y

    def get_K(self, p, X, Y):
        p = np.split(list(p), self.cumul_dims)
        res = np.empty([self.n_tasks, self.n_tasks]).tolist()
        for i in range(self.n_tasks):
            for j in range(self.n_tasks):
                if i == j:
                    X_, Y_ = X[i], Y[j]
                    res[i][j] = self.K[i][j].get_K(p[i], X_, Y_)

                elif i < j:
                    X_, Y_ = self._get_XY(X[i], Y[j])
                    res[i][j] = self.K[i][j](flatten([p[i], p[j]]), X_, Y_)
                else:
                    X_, Y_ = self._get_XY(X[i], Y[j])
                    res[i][j] = self.K[j][i](flatten([p[j], p[i]]), Y_, X_)

        res = np.block(res)
        return res

    def get_dK_dY(self, p, X, Y):
        p = np.split(list(p), self.cumul_dims)
        res = np.empty([self.n_tasks, self.n_tasks]).tolist()
        for i in range(self.n_tasks):
            for j in range(self.n_tasks):
                X_, Y_ = self._get_XY(X[i], Y[j])
                if i == j:
                    res[i][j] = self.dK_dY_[i][j](p[i], X_, Y_)

                elif i < j:
                    res[i][j] = self.dK_dY_[i][j](flatten([p[i], p[j]]), X_, Y_)

                else:
                    res[i][j] = self.dK_dY_[j][i](flatten([p[i], p[j]]), X_, Y_)

        res = -np.block(res)
        return res

    def get_d2K_dY2(self, p, X, Y):
        p = np.split(list(p), self.cumul_dims)
        res = np.empty([self.n_tasks, self.n_tasks]).tolist()
        for i in range(self.n_tasks):
            for j in range(self.n_tasks):
                X_, Y_ = self._get_XY(X[i], Y[j])
                if i == j:
                    res[i][j] = self.d2K_dY2_[i][j](p[i], X_, Y_)
                elif i < j:
                    res[i][j] = self.d2K_dY2_[i][j](flatten([p[i], p[j]]), X_, Y_)
                else:
                    res[i][j] = self.d2K_dY2_[j][i](flatten([p[j], p[i]]), Y_, X_)

        res = np.block(res)

        return res

    def get_d2K_dXdY(self, p, X, Y):
        p = np.split(list(p), self.cumul_dims)
        res = np.empty([self.n_tasks, self.n_tasks]).tolist()
        for i in range(self.n_tasks):
            for j in range(self.n_tasks):
                X_, Y_ = self._get_XY(X[i], Y[j])
                if i == j:
                    res[i][j] = self.d2K_dXdY_[i][j](p[i], X_, Y_)
                elif i < j:
                    res[i][j] = self.d2K_dXdY_[i][j](flatten([p[i], p[j]]), X_, Y_)
                else:
                    res[i][j] = self.d2K_dXdY_[j][i](flatten([p[j], p[i]]), Y_, X_)

        res = np.block(res)
        return res

    def get_d4K_dX2dY2(self, p, X, Y):
        p = np.split(list(p), self.cumul_dims)
        res = np.empty([self.n_tasks, self.n_tasks]).tolist()
        for i in range(self.n_tasks):
            for j in range(self.n_tasks):
                X_, Y_ = self._get_XY(X[i], Y[j])
                if i == j:
                    res[i][j] = self.d4K_dX2dY2_[i][j](p[i], X_, Y_)
                elif i < j:
                    res[i][j] = self.d4K_dX2dY2_[i][j](flatten([p[i], p[j]]), X_, Y_)
                else:
                    res[i][j] = self.d4K_dX2dY2_[j][i](flatten([p[j], p[i]]), Y_, X_)

        res = np.block(res)
        return res


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

    for i in range(gp_info["n_tasks"]):
        gp_info_i = gp_info["func_" + str(i + 1)]
        for j, par in enumerate(gp_info_i["hyperpars"]):
            # --
            sampling_info["hyp_sampled"].append(par)
            # --
            sampling_info["ranges"] = np.vstack(
                (
                    sampling_info["ranges"],
                    np.array(gp_info_i["hyperpars"][par]["range"]),
                )
            )
            # --
            if "ini" in gp_info_i["hyperpars"][par]:
                sampling_info["inis"].append(gp_info_i["hyperpars"][par]["ini"])
            else:
                sampling_info["inis"].append(
                    (
                        gp_info_i["hyperpars"][par]["range"][1]
                        - gp_info_i["hyperpars"][par]["range"][0]
                    )
                    / 2.0
                )
            # --
            if "sampling_invfunc" in gp_info_i["hyperpars"][par]:
                sampling_info["sampling_invfuncs"].append(
                    gp_info_i["hyperpars"][par]["sampling_invfunc"]
                )
            else:
                sampling_info["sampling_invfuncs"].append(lambda x: x)

    return sampling_info


def get_reconstruction(x, gp, gp_info):
    mean, cov = gp.predict(x, predict_cov=True)
    res = {"mean": mean, "cov": cov}

    derivatives = False
    for task in gp_info.keys():
        if isinstance(gp_info[task], dict):
            if "derivatives" in gp_info[task].keys():
                derivatives = True

    if derivatives:
        mean_d1, cov_d1 = gp.predict_d1(x, predict_cov=True)
        # mean_d2, cov_d2 = gp.predict_d2(x, predict_cov=True)
        cov_01 = gp.predict_cov_01(x)
        res.update(
            {
                "mean_d1": mean_d1,
                "cov_d1": cov_d1,
                # "mean_d2": mean_d2,
                # "cov_d2": cov_d2,
                "cov_01": cov_01,
            }
        )

    for i in range(gp_info["n_tasks"]):
        name_func = gp_info["func_" + str(i + 1)]["name_function"]
        res[name_func] = {
            "x": x[i],
            "mean": mean[i * len(x[i]) : (i + 1) * len(x[i])],
            "err": np.sqrt(
                np.diag(
                    cov[
                        i * len(x[i]) : (i + 1) * len(x[i]),
                        i * len(x[i]) : (i + 1) * len(x[i]),
                    ]
                )
            ),
            "cov": cov[
                i * len(x[i]) : (i + 1) * len(x[i]),
                i * len(x[i]) : (i + 1) * len(x[i]),
            ],
        }
        if derivatives:
            res[name_func].update(
                {
                    "mean_d1": mean_d1[i * len(x[i]) : (i + 1) * len(x[i])],
                    "err_d1": np.sqrt(
                        np.diag(
                            cov_d1[
                                i * len(x[i]) : (i + 1) * len(x[i]),
                                i * len(x[i]) : (i + 1) * len(x[i]),
                            ]
                        )
                    ),
                    "cov_d1": cov_d1[
                        i * len(x[i]) : (i + 1) * len(x[i]),
                        i * len(x[i]) : (i + 1) * len(x[i]),
                    ],
                    # "mean_d2": mean_d2[i * len(x[i]) : (i + 1) * len(x[i])],
                    # "err_d2": np.sqrt(
                    #     np.diag(
                    #         cov_d2[
                    #             i * len(x[i]) : (i + 1) * len(x[i]),
                    #             i * len(x[i]) : (i + 1) * len(x[i]),
                    #         ]
                    #     )
                    # ),
                    # "cov_d2": cov_d2[
                    #     i * len(x[i]) : (i + 1) * len(x[i]),
                    #     i * len(x[i]) : (i + 1) * len(x[i]),
                    # ],
                    "cov_01": cov_01[
                        i * len(x[i]) : (i + 1) * len(x[i]),
                        i * len(x[i]) : (i + 1) * len(x[i]),
                    ],
                }
            )

    return res


def merge_info(gp_info):
    mtgp_info = {}
    for i, dic in enumerate(gp_info):
        mtgp_info["func_" + str(i + 1)] = dic
    mtgp_info["n_tasks"] = len(gp_info)
    return mtgp_info


def merge_data(data):
    mtgp_data = {
        "x": [],
        "y": [],
        "err": [],
        "cov": [],
    }
    for i, dic in enumerate(data):
        mtgp_data["x"].append(dic["x"])
        mtgp_data["y"].append(dic["y"])
        mtgp_data["err"].append(dic["err"])
        mtgp_data["cov"] = block_diag(mtgp_data["cov"], dic["cov"])

    mtgp_data["cov"] = mtgp_data["cov"][1:]

    return mtgp_data


################################################################################
### Functions for master.py
################################################################################
def get_covfunc(gp_info):
    return Covfunc(gp_info)


def get_mean_priors(gp_info, data):
    """Mean priors on the reconstruction getter"""

    null_func = lambda x: np.zeros(len(x))
    priors = {
        "mean": [null_func for i in range(gp_info["n_tasks"])],
        "mean_d1": [null_func for i in range(gp_info["n_tasks"])],
        "mean_d2": [null_func for i in range(gp_info["n_tasks"])],
    }

    names = ["mean", "mean_d1", "mean_d2"]
    for name in names:
        for i in range(gp_info["n_tasks"]):
            if "mean_priors" in gp_info["func_" + str(i + 1)]:
                if name in gp_info["func_" + str(i + 1)]["mean_priors"]:
                    priors[name][i] = gp_info["func_" + str(i + 1)]["mean_priors"][name]

    def func(name, x):
        res = []
        for i in range(gp_info["n_tasks"]):
            res.append(priors["mean"][i](x[i]))
        return np.block(res)

    Y = np.block(data["y"]) - np.block(
        [priors["mean"][i](data["x"][i]) for i in range(gp_info["n_tasks"])]
    )

    res = {
        "Y": Y,
        "YT": np.transpose(Y),
        "mean": lambda x: func("mean", x),
        "mean_d1": lambda x: func("mean_d1", x),
        "mean_d2": lambda x: func("mean_d2", x),
    }

    return res


def get_n_data(data):
    """Returns the number of data"""
    ct = 0
    for i in range(np.shape(data["x"])[0]):
        ct += len(data["x"][i])
    return ct
