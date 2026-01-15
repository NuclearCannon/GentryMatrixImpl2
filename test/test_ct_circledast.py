import numpy as np
import sys
import os
import cProfile

current_dir = os.path.dirname(os.path.abspath(__file__))
utils_dir = os.path.join(current_dir, "../")

sys.path.append(utils_dir)

import src
from src import param_gen, fhe, polynomial, utils
from src.encode import Encoder

# 相比于test_all.py，本文件不含编码解码部分，
# 单纯描述噪声增长程度

def unify(a, q):
    a = a % q
    a[a>(q//2)] -= q
    return a

n=32
p=3
q = param_gen.param_find_q(n,p,2_000_000_000)
params = {"n": n, "p":p, "q":q}
print(params)

# 生成密钥
h = n//2
sk = polynomial.poly_sk(n,p,q,h)
ksks = fhe.create_ksks_for_circledast(sk, n, p, q)


poly1 = polynomial.poly_coeff_bound(n,p,q,10)
# poly1 = polynomial.ZqiXYW(n,p,q,zero=True)
poly2 = polynomial.poly_coeff_bound(n,p,q,10)
# poly2 = polynomial.ZqiXYW(n,p,q,zero=True)


ct1 = fhe.encrypt_XYW(poly1, sk, no_error=True) # 我们不关心加密本身造成的误差
ct2 = fhe.encrypt_XYW(poly2, sk, no_error=True) # 

pr = cProfile.Profile()
pr.enable()
ct3 = fhe.circledast_ct(ct1, ct2, ksks)
pr.disable()
pr.dump_stats('output.prof')  # 保存为 .prof 文件

poly3 = fhe.decrypt_XYW(ct3[0], ct3[1], sk)
poly3_pt = poly1.circledast(poly2)

error = poly3 - poly3_pt

error = error.to_coeff()
error %= q
error[error>(q//2)] -= q

print(f"error=\n{error[0,:,:,0]}")
error = np.max(np.abs(error))

print(f"error={error}")



