# 本文件意在测试KeySwitch的误差程度，以及之前所有内容的正确性
import numpy as np
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
utils_dir = os.path.join(current_dir, "../")

sys.path.append(utils_dir)

import src
from src import param_gen, fhe, polynomial, utils
from src.encode import Encoder




n=8
p=3
q = param_gen.param_find_q(n,p,2_000_000_000)
params = {"n": n, "p":p, "q":q}
print(params)


delta = 1000

encoder = Encoder(n,p,q)

mat = utils.distribution.random_complex_array((n,n,p-1), 1)
mat_sq = mat * mat

poly = encoder.encode_to_polynomial(mat, delta)
poly_sq = poly * poly

mat_dec = encoder.decode_from_polynomial(poly_sq) / (delta ** 2)

error = mat_dec - mat_sq

print("error", np.max(np.abs(error)))


