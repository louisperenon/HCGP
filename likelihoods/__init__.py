__all__ = ["log_marginal"]


def get_likelihood(name):

    if name == "log_marginal":
        from log_marginal import LogLikelihood

    return LogLikelihood
