from .SquaredExponential import SquaredExponential
from .RationalQuadratic import RationalQuadratic
from .Matern32 import Matern32
from .Matern52 import Matern52
from .Matern72 import Matern72
from .Matern92 import Matern92


def get_convolution(K_1, K_2):
    name_1 = K_1.name
    name_2 = K_2.name

    sp_hyps_1 = K_1.sp_hyps
    sp_hyps_2 = K_2.sp_hyps

    if name_1 == name_2 == "SquaredExponential":
        from .SquaredExponential_SquaredExponential import Convolution

    elif name_1 == name_2 == "Matern32":
        from .Matern32_Matern32 import Convolution
    else:
        raise ValueError("Convolution not implemented yet")

    return Convolution(sp_hyps_1, sp_hyps_2)
