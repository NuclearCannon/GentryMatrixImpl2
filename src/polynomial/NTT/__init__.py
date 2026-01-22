import numpy as np
from ...param_gen import find_zeta, find_eta
from ... import utils

from functools import lru_cache
# 本模块的：
from .standard import ntt_standard
from .rader import ntt_rader
from .ntt_w import ntt_W, intt_W
from .ntt_i import ntt_mathcal_I, intt_mathcal_I, get_mathcal_I