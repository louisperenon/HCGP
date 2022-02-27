import numpy as np
from likelihoods.__init__ import get_likelihood
from explorers.__init__ import get_minimiser
from explorers.__init__ import get_sampler

# from plotter import plot_reconstruction
from defaults import *
from gp import GP


################################################################################
###                            The Master Engine                             ###
################################################################################
class Engine:
    """ ... """  # docstring has to be written in the future

    def __init__(self, run_options, gp_info, data):

        # --- Setting the missing info to the defaults values
        self.set_defaults(run_options, gp_info)

        # --- Setting the gp
        self.gp_info = gp_info
        if "n_tasks" not in gp_info.keys():
            import single_task as mode
        else:
            import multi_task as mode
        self.gp_info["mode"] = mode
        self.gp = GP(gp_info, data)

        # --- Setting the likelihood instance
        self.loglike = get_likelihood(run_options["likelihood"])(self.gp)

        # --- Running the choices
        if not "plot" in run_options["method"]:
            self.sampling_info = self.set_sampling_info(
                run_options,
                gp_info,
            )
            self.minimiser = get_minimiser(self.sampling_info["minimiser"])
            if "optimisation" in run_options["method"]:
                self.run_minimise()
                if run_options["print_plots"]:
                    plot_reconstruction(
                        data,
                        self.get_optimised_reconstruction(),
                        run_options["name_run"],
                        "optimised",
                    )

        # # --- Progress checkers

    ###############
    ### Setters ###
    ###############

    ###
    def set_defaults(self, run_options, gp_info):
        keys = run_options.keys()
        if not "method" in keys:
            run_options["method"] = run_options_method
        if not "print_plots" in keys:
            run_options["print_plots"] = run_options_print_plots
        if not "likelihood" in keys:
            run_options["likelihood"] = run_options_likelihood
        if not "minimiser" in keys:
            run_options["minimiser"] = run_options_minimiser
        if not "sampler" in keys:
            run_options["sampler"] = run_options_sampler

        keys = gp_info.keys()
        if not "matrix_inversion_method" in keys:
            gp_info["matrix_inversion_method"] = gp_info_matrix_inversion_method
        if not "derivatives" in keys:
            gp_info["derivatives"] = gp_info_derivatives
        if not "gradient" in keys:
            gp_info["gradient"] = gp_info_gradient

    ###
    def set_sampling_info(self, run_options, gp_info):
        sampling_info = {
            "likelihood": run_options["likelihood"],
            "minimiser": run_options["minimiser"],
            "sampler": run_options["sampler"],
        }
        sampling_info.update(gp_info["mode"].get_sampling_info(gp_info))
        return sampling_info

    ###
    def compute_mlikelihood(self, p):
        if not np.all(
            (self.sampling_info["ranges"][:, 0] <= p)
            & (p <= self.sampling_info["ranges"][:, 1])
        ):
            return np.inf

        pars = [f(a) for f, a in zip(self.sampling_info["sampling_invfuncs"], p)]
        pars = dict(zip(self.sampling_info["hyp_sampled"], pars))

        try:
            res = -self.loglike.value(pars)
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
        res = self.gp_info["mode"].get_reconstruction(x, self.gp, self.gp_info)
        return res

    ###
    def get_minimisation_results(self):
        res = {
            "bestfit": self.bestfit,
            "bestsamp": self.bestsamp,
            "maxlnl": self.maxlnl,
        }
        return res

    ###############
    ### Runners ###
    ###############
    def run_minimise(self):

        run = self.minimiser(self.compute_mlikelihood, self.sampling_info)
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
