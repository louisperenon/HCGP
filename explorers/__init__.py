def get_minimiser(name):

    if name == "differential_evolution":
        from .differential_evolution import run

    return run


def get_sampler(name):
    pass

