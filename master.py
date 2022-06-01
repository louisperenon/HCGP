import sys
import numpy as np

# from plotter import plot_reconstruction
from .defaults import *
from .explorers.__init__ import get_minimiser, get_sampler
from .gp import GP


################################################################################
###                            The Master Engine                             ###
################################################################################
class Engine:
    """ ... """  # TODO docstring has to be written in the future

    def __init__(self, run_options, gp_info, data):

        self.run_options = run_options
        self.gp_info = gp_info

        # --- Setting the missing info to the defaults values
        self.set_defaults()

        # --- Setting the gp
        if "n_tasks" not in gp_info.keys():
            import HCGP.single_task as mode
        else:
            import HCGP.multi_task as mode
        self.gp_info["mode"] = mode
        self.gp = GP(self.gp_info, data)

        # --- Running the choices
        if not "plot" in run_options["method"]:
            self.sampling_info = self.set_sampling_info()
            if "optimisation" in run_options["method"]:
                self.run_minimise()
                if run_options["print_plots"]:
                    plot_reconstruction(
                        data,
                        self.get_optimised_reconstruction(),
                        run_options["name_run"],
                        "optimised",
                    )

    ###############
    ### Setters ###
    ###############

    ###
    def set_defaults(self):  # TODO Make it a function
        """Sets the missing keys in run_options and gp_info to the default values"""
        keys = self.run_options.keys()
        if not "method" in keys:
            self.run_options["method"] = run_options_method
        if not "print_plots" in keys:
            self.run_options["print_plots"] = run_options_print_plots
        if not "likelihood" in keys:
            self.run_options["likelihood"] = run_options_likelihood
        if not "minimiser" in keys:
            self.run_options["minimiser"] = run_options_minimiser
        if not "sampler" in keys:
            self.run_options["sampler"] = run_options_sampler

        keys = self.gp_info.keys()
        if not "matrix_inversion_method" in keys:
            self.gp_info["matrix_inversion_method"] = gp_info_matrix_inversion_method
        if not "derivatives" in keys:
            self.gp_info["derivatives"] = gp_info_derivatives
        if not "gradient" in keys:
            self.gp_info["gradient"] = gp_info_gradient

    ###
    def set_sampling_info(self):  # TODO Make it a function
        """Sets the sampling_info dict"""
        sampling_info = {
            "likelihood": self.run_options["likelihood"],
            "minimiser": self.run_options["minimiser"],
            "sampler": self.run_options["sampler"],
        }
        sampling_info.update(self.gp_info["mode"].get_sampling_info(self.gp_info))
        return sampling_info

    ###
    def compute_mlikelihood(self, p):
        """Compute the -1 * the log marginal likelihood of the GP"""
        if not np.all(
            (self.sampling_info["ranges"][:, 0] <= p)
            & (p <= self.sampling_info["ranges"][:, 1])
        ):
            return np.inf

        pars = [f(a) for f, a in zip(self.sampling_info["sampling_invfuncs"], p)]
        pars = dict(zip(self.sampling_info["hyp_sampled"], pars))

        try:
            self.gp.set_hyp_values(pars)
            res = -self.gp.log_marginal_likelihood()
        except:
            return np.inf

        if np.isnan(res):
            return np.inf

        return res

    ###############
    ### Getters ###
    ###############

    ###
    def get_optimised_reconstruction(self, x):
        """Return the dictionary of the optimised restruction"""
        res = self.gp_info["mode"].get_reconstruction(x, self.gp, self.gp_info)
        return res

    ###
    def get_minimisation_results(self):
        """Returns the optimisation results"""
        res = {
            "bestfit": self.bestfit,
            "bestsamp": self.bestsamp,
            "maxlnl": self.maxlnl,
        }
        return res

    ###############
    ### Runners ###
    ###############
    def run_minimise(self):  # TODO Make it a function
        """Runs the minimisation given the chosen minimiser chosen"""
        minimiser = get_minimiser(self.sampling_info["minimiser"])
        run = minimiser(self.compute_mlikelihood, self.sampling_info)
        self.bestfit = run["bestfit"]
        self.bestsamp = run["bestsamp"]
        self.maxlnl = self.gp.log_marginal_likelihood()

        print("")
        print("*** Minimisation results ***")
        print("")
        for par in self.sampling_info["hyp_sampled"]:
            print(
                "> %s: bestfit = %s; bestsamp = %s "
                % (par, self.bestfit[par], self.bestsamp[par])
            )
        print("> maxlnl = %s" % self.maxlnl)
        print("")
