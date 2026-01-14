import numpy as np
from ..polynomial import ZqiXYW, poly_dg, poly_uniform, poly_coeff_bound


def encrypt_XYW(message: ZqiXYW, sk: ZqiXYW, **kwargs):
    n,p,q = message.npq()
    assert sk.npq() == (n,p,q)

    if kwargs.get("no_a_part", False):
        a = ZqiXYW(n,p,q,zero=True)
    else:
        a = poly_uniform(n,p,q)
    
    if kwargs.get("no_error", False):
        e = ZqiXYW(n,p,q,zero=True)
    else:
        e = poly_dg(n,p,q, sigma=kwargs.get("sigma", 3.19))

    b = message - a * sk + e

    return a, b

def decrypt_XYW(ct_a, ct_b, sk):
    n,p,q = ct_a.npq()
    assert ct_b.npq() == (n,p,q)
    assert sk.npq() == (n,p,q)
    m = ct_a*sk+ct_b
    return m

