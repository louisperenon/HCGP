from __future__ import absolute_import
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy import misc
import classy

from master import Engine
from kernels import SquaredExponential
import multi_task as mtgp


##############
### Inputs ###
##############

pred_z = np.linspace(0, 10, 200)
data_a_z = np.linspace(1, 9, 30)
data_b_z = np.linspace(1, 9, 30)


#################
### GP config ###
#################
run_options = {
    "name_run": "test",
    "method": ["optimisation"],
}


gp_info_a = {
    "name_function": "a",
    "label_function": r"$a$",
    "label_x": r"$z$",
    "kernel": SquaredExponential(sigma="sigma_a", xi="xi_a"),
    "hyperpars": {
        "sigma_a": {
            "label": r"\sigma",
            "range": [-5, 5],
            "sampling_invfunc": lambda x: 10 ** np.array(x),
        },
        "xi_a": {
            "label": r"\xi",
            "range": [-5, 5],
            "sampling_invfunc": lambda x: 10 ** np.array(x),
        },
    },
    "mean_priors": {
        "mean": lambda x: 0 * x,
        "mean_d1": lambda x: 0 * x,
    },
    "derivatives": True,
}

gp_info_b = {
    "name_function": "b",
    "label_function": r"$b$",
    "label_x": r"$z$",
    "kernel": SquaredExponential(sigma="sigma_b", xi="xi_b"),
    "hyperpars": {
        "sigma_b": {
            "label": r"\sigma",
            "range": [-5, 5],
            "sampling_invfunc": lambda x: 10 ** np.array(x),
        },
        "xi_b": {
            "label": r"\xi",
            "range": [-5, 5],
            "sampling_invfunc": lambda x: 10 ** np.array(x),
        },
    },
    "mean_priors": {
        "mean": lambda x: 0 * x,
        "mean_d1": lambda x: 0 * x,
    },
    "derivatives": True,
}

############
### Data ###
############
def data_func_a(x):
    return 4 * x ** 2 + 3


def data_func_b(x):
    return 2 * x ** 2 - 5


def make_mock(z, func):
    mean = func(z)
    err = 50 * np.ones(len(z))
    cov = np.diag(err ** 2)
    return {
        "x": np.array(z),
        "y": np.random.multivariate_normal(mean, cov),
        "err": err,
        "cov": cov,
    }


data_a = make_mock(data_a_z, data_func_a)
data_b = make_mock(data_b_z, data_func_b)


#################
### MTGP COMP ###
#################
mtgp_info = mtgp.merge_info([gp_info_a, gp_info_b])
mtgp_data = mtgp.merge_data([data_a, data_b])

results_mtgp = Engine(run_options, mtgp_info, mtgp_data)
recon_mtgp = results_mtgp.get_optimised_reconstruction([pred_z, pred_z])


#################
### STGP COMP ###
#################
results_a = Engine(run_options, gp_info_a, data_a)
recon_a = results_a.get_optimised_reconstruction(pred_z)

results_b = Engine(run_options, gp_info_b, data_b)
recon_b = results_b.get_optimised_reconstruction(pred_z)


############
### Plot ###
############
plt.figure(figsize=(16, 6))
color_data = "black"
color_stgp = "blue"
color_stgpd = "green"
color_mtgp = "purple"
color_mtgpd = "red"

###
plt.subplot(121)
plt.xlabel(mtgp_info["func_1"]["label_x"], fontsize=26)
plt.ylabel(mtgp_info["func_1"]["label_function"], fontsize=26)
# data
plt.errorbar(
    data_a["x"],
    data_a["y"],
    data_a["err"],
    fmt=".k",
    capsize=1,
    label="Data",
    color=color_data,
)
# stgp
plt.plot(
    recon_a["x"],
    recon_a["mean"],
    color=color_stgp,
    label="STGP",
)
plt.fill_between(
    recon_a["x"],
    recon_a["mean"] - recon_a["err"],
    recon_a["mean"] + recon_a["err"],
    color=color_stgp,
    label="68% C.I.",
    alpha=0.4,
)
# stgpd
plt.plot(
    recon_a["x"],
    recon_a["mean_d1"],
    color=color_stgpd,
    label="STGP'",
)
plt.fill_between(
    recon_a["x"],
    recon_a["mean_d1"] - recon_a["err_d1"],
    recon_a["mean_d1"] + recon_a["err_d1"],
    color=color_stgpd,
    alpha=0.4,
)
# mtgp
plt.plot(
    recon_mtgp["a"]["x"],
    recon_mtgp["a"]["mean"],
    color=color_mtgp,
    label="MTGP",
)
plt.fill_between(
    recon_mtgp["a"]["x"],
    recon_mtgp["a"]["mean"] - recon_mtgp["a"]["err"],
    recon_mtgp["a"]["mean"] + recon_mtgp["a"]["err"],
    color=color_mtgp,
    alpha=0.4,
)
# mtgpd
plt.plot(
    recon_mtgp["a"]["x"],
    recon_mtgp["a"]["mean_d1"],
    color=color_mtgpd,
    label="MTGP'",
)
plt.fill_between(
    recon_mtgp["a"]["x"],
    recon_mtgp["a"]["mean_d1"] - recon_mtgp["a"]["err_d1"],
    recon_mtgp["a"]["mean_d1"] + recon_mtgp["a"]["err_d1"],
    color=color_mtgpd,
    alpha=0.4,
)
plt.yticks(fontsize=18)
plt.xticks(fontsize=18)
plt.legend(fontsize=16)


####
plt.subplot(122)
plt.xlabel(mtgp_info["func_2"]["label_x"], fontsize=26)
plt.ylabel(mtgp_info["func_2"]["label_function"], fontsize=26)
# data
plt.errorbar(
    data_b["x"],
    data_b["y"],
    data_b["err"],
    fmt=".k",
    capsize=1,
    label="Data",
    color=color_data,
)
# stgp
plt.plot(
    recon_b["x"],
    recon_b["mean"],
    color=color_stgp,
    label="STGP",
)
plt.fill_between(
    recon_b["x"],
    recon_b["mean"] - recon_b["err"],
    recon_b["mean"] + recon_b["err"],
    color=color_stgp,
    alpha=0.4,
)
# stgpd
plt.plot(
    recon_b["x"],
    recon_b["mean_d1"],
    color=color_stgpd,
    label="STGP'",
)
plt.fill_between(
    recon_b["x"],
    recon_b["mean_d1"] - recon_b["err_d1"],
    recon_b["mean_d1"] + recon_b["err_d1"],
    color=color_stgpd,
    alpha=0.4,
)
# mtgp
plt.plot(
    recon_mtgp["b"]["x"],
    recon_mtgp["b"]["mean"],
    color=color_mtgp,
    label="MTGP",
)
plt.fill_between(
    recon_mtgp["b"]["x"],
    recon_mtgp["b"]["mean"] - recon_mtgp["b"]["err"],
    recon_mtgp["b"]["mean"] + recon_mtgp["b"]["err"],
    color=color_mtgp,
    alpha=0.4,
)
# mtgpd
plt.plot(
    recon_mtgp["b"]["x"],
    recon_mtgp["b"]["mean_d1"],
    color=color_mtgpd,
    label="MTGP'",
)
plt.fill_between(
    recon_mtgp["b"]["x"],
    recon_mtgp["b"]["mean_d1"] - recon_mtgp["b"]["err_d1"],
    recon_mtgp["b"]["mean_d1"] + recon_mtgp["b"]["err_d1"],
    color=color_mtgpd,
    alpha=0.4,
)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.tight_layout()
plt.show()
plt.clf()
