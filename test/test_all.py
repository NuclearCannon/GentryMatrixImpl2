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

# 生成密钥
h = n//2
sk = polynomial.poly_sk(n,p,q,h)
ksks = fhe.create_ksks_for_circledast(sk, n, p, q)

delta = 10000

encoder = Encoder(n,p,q)

mat1 = utils.distribution.random_complex_array((n,n,p-1), 10)
poly1 = encoder.encode_to_polynomial(mat1, delta)
mat2 = utils.distribution.random_complex_array((n,n,p-1), 10)
poly2 = encoder.encode_to_polynomial(mat2, delta)

ct1 = fhe.encrypt_XYW(poly1, sk)
ct2 = fhe.encrypt_XYW(poly2, sk)

ct3 = fhe.circledast_ct(ct1, ct2, ksks)

poly3 = fhe.decrypt_XYW(ct3[0], ct3[1], sk)
# circledast得到的是A@B.conj().T/n，需要补一个n才能得到乘法结果
mat3 = encoder.decode_from_polynomial(poly3) / (delta ** 2) * n

mat3_real = np.einsum('ijm,kjm->ikm', mat1, mat2.conj(), optimize=True)

error = mat3 - mat3_real

error = np.max(np.abs(error))

print(f"error={error}")





