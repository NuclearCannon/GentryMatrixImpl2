# 本文件意在测试KeySwitch的误差程度，以及之前所有内容的正确性
import numpy as np
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
utils_dir = os.path.join(current_dir, "../")

sys.path.append(utils_dir)

import src
from src import param_gen, fhe, polynomial
from src.utils.distribution import generate_uniform, generate_hwt
from src.fhe import KeySwitchingKey




n=8
p=3
q = param_gen.param_find_q(n,p,2_000_000_000)
params = {"n": n, "p":p, "q":q}
print(params)


# 生成密钥
h = n
sk = polynomial.poly_sk(n,p,q,h)
# sk2 = polynomial.poly_hwt(n,p,q,h)
sk2 = polynomial.poly_sk(n,p,q,h)



ksk = KeySwitchingKey(
    sk_from=sk2,
    sk_to=sk,
    params=params,
    no_error_ksk=False,
    no_a_part_ksk=False
)
# 理论上，ksk中的cts应该是sk2在sk下的模Q密文。让我们打开它看看。

Q = ksk.Q
B = ksk.B
skQ = sk.switch_q(Q)
sk2_raw = sk2.to_coeff()

for i,(a,b) in enumerate(ksk.cts):
    m = fhe.encrypt.decrypt_XYW(a,b,skQ).to_coeff()
    m %= Q
    m[m>Q//2] -= Q
    sk2QB = sk2_raw * B**i * Q // q

    error = m - sk2QB
    error %= Q
    error[error>Q//2] -= Q
    print(f"at cts[{i}], error={np.max(np.abs(error))}")    # 十几的程度是正常的




# 一个明文信息
m = polynomial.poly_coeff_bound(n,p,q,2000)
a,b = fhe.encrypt.encrypt_XYW(m, sk2, no_error=False, no_a_part=False)

# 分别用两种KS方案进行KS

a2, b2 = ksk.key_switch_big(a, b)

# 解密出结果
m1 = fhe.encrypt.decrypt_XYW(a, b, sk2) - m    
m2 = fhe.encrypt.decrypt_XYW(a2, b2, sk) - m

m1 = m1.to_coeff()
m2 = m2.to_coeff()

m1 %= q
m1[m1>(q//2)] -= q

m2 %= q
m2[m2>(q//2)] -= q

print("np.max(np.abs(m1))")
print(np.max(np.abs(m1)))
print("np.max(np.abs(m2))")
print(np.max(np.abs(m2)))





