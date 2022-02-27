from .SquaredExponential import SquaredExponential
from .Matern32 import Matern32


def get_convolution(K_1, K_2):
    name_1 = K_1.name
    name_2 = K_2.name

    sp_hyps_1 = K_1.sp_hyps
    sp_hyps_2 = K_2.sp_hyps

    if name_1 == name_2 == "SquaredExponential":
        from .SquaredExponential_SquaredExponential import Convolution

    else:
        raise ValueError("Convolution not implemented yet")

    return Convolution(sp_hyps_1, sp_hyps_2)
