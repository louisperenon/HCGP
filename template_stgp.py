import matplotlib.pyplot as plt
import numpy as np

from kernels import SquaredExponential
from master import Engine

##############
### Inputs ###
##############

pred_z = np.linspace(0, 10, 100)
data_z = np.linspace(1, 9, 50)


#################
### GP config ###
#################

run_options = {
    "name_run": "test",
    "method": ["optimisation"],
    # "print_chains": True,
    "print_plots": False,
}


gp_info = {
    "name_function": "g",
    "label_function": r"$g$",
    "label_x": r"$z$",
    "kernel": SquaredExponential(sigma="sigma1", xi="xi1"),
    "hyperpars": {
        "sigma1": {
            "label": r"\sigma",
            "range": [-5, 5],
            "sampling_invfunc": lambda x: 10 ** np.array(x),
        },
        "xi1": {
            "label": r"\xi",
            "range": [-5, 5],
            "sampling_invfunc": lambda x: 10 ** np.array(x),
        },
    },
    "mean_priors": {
        "mean": lambda x: 0 * x,
    },
    "derivatives": True,
}


############
### Data ###
############
def data_func(x):
    """Function aroudn which the mock data is generated"""
    return 10 * x ** 2


def make_mock(z, rel_err, func):
    """Creates mock data realistations"""
    mean = func(z)
    err = 100 * np.ones(len(z))
    cov = np.diag(err ** 2)
    return {
        "x": z,
        "y": np.random.multivariate_normal(mean, cov),
        "err": err,
        "cov": cov,
    }


data = make_mock(data_z, 0.1, data_func)


####################
### Computations ###
####################

results = Engine(run_options, gp_info, data)
reco = results.get_optimised_reconstruction(pred_z)

#############
### Plots ###
#############

color_data = "black"
color_gp = "blue"
color_gpd = "purple"
color_gpd2 = "darkred"

### Plot
plt.figure(figsize=(8, 6))
plt.xlabel(gp_info["label_x"], fontsize=26)
plt.ylabel(gp_info["label_function"], fontsize=26)
# data
plt.errorbar(
    data["x"],
    data["y"],
    data["err"],
    fmt=".",
    capsize=1,
    label="Data",
    color=color_data,
)
# gp
plt.plot(
    reco["x"],
    reco["mean"],
    color=color_gp,
    label="GP",
)
plt.fill_between(
    reco["x"],
    reco["mean"] - reco["err"],
    reco["mean"] + reco["err"],
    color=color_gp,
    alpha=0.4,
    label="68% C.I.",
)
# First derivative
plt.plot(
    reco["x"],
    reco["mean_d1"],
    color=color_gpd,
    label="GP'",
)
plt.fill_between(
    reco["x"],
    reco["mean_d1"] - reco["err_d1"],
    reco["mean_d1"] + reco["err_d1"],
    color=color_gpd,
    alpha=0.4,
)
# Second derivative
plt.plot(
    reco["x"],
    reco["mean_d2"],
    color=color_gpd2,
    label="GP''",
)
plt.fill_between(
    reco["x"],
    reco["mean_d2"] - reco["err_d2"],
    reco["mean_d2"] + reco["err_d2"],
    color=color_gpd2,
    alpha=0.4,
)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(fontsize=16)
plt.tight_layout()
plt.show()
plt.clf()
