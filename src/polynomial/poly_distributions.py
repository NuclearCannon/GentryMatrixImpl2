from ..utils import distribution
from . import ZqiXYW
import numpy as np

def poly_sk(n, p, q, h):
    """创建一个在Y^0分量上呈hwt分布的多项式对象"""
    r = np.zeros((2,p-1,n,n), dtype=object)
    r[:,:,:,0] = distribution.generate_hwt((2, p-1, n),h)
    return ZqiXYW(n, p, q, coeff=r)


def poly_hwt(n, p, q, h):
    """创建一个呈hwt分布的多项式对象，不只是在Y^0分量上"""
    r = distribution.generate_hwt((2, p-1, n, n),h)
    return ZqiXYW(n, p, q, coeff=r)

def poly_uniform(n,p,q):
    assert isinstance(q, int)
    return ZqiXYW(n,p,q,uniform=True)

def poly_coeff_bound(n,p,q,B):
    r = distribution.generate_uniform((2, p-1, n, n),2*B)-B
    return ZqiXYW(n, p, q, coeff=r)


def poly_dg(n, p, q, sigma=3.19):
    """创建一个离散高斯分布的多项式对象"""
    r = distribution.generate_discrete_gaussian((2, p-1, n, n),sigma)
    return ZqiXYW(n, p, q, coeff=r)