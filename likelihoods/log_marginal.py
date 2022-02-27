class LogLikelihood:
    """Eats a gp.Engine() instance"""

    def __init__(self, gp):
        self.gp = gp

    def value(self, p):
        self.gp.set_hyp_values(p)
        return self.gp.log_marginal_likelihood()

    def value_grad(self, p):
        self.gp.set_hyp_values(p)
        return self.gp.grad_log_marginal_likelihood()
