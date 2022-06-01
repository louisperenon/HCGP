from scipy import optimize

"""This is the minimiser package for the GP's
"""


def run(function, sampling_info):

    res = optimize.differential_evolution(
        function,
        sampling_info["ranges"],
        # tol=1e-6,
        tol=1e-12,
        polish=True,
        disp=True,
        maxiter=20000,
    )

    bestfit = {}
    bestsamp = {}
    for i, par in enumerate(sampling_info["hyp_sampled"]):
        bestsamp[par] = res.x[i]
        bestfit[par] = sampling_info["sampling_invfuncs"][i](res.x[i])

    results = {"bestsamp": bestsamp, "bestfit": bestfit}
    # print("hello", results)

    return results
