from . import find_primitive_root, ntt_standard, find_eta

from . import utils

import numpy as np


def convolve_standard(a, b, q):
    """基于ntt_standard的卷积函数设计"""
    n = len(a)
    assert len(b) == n, "两个参数的长度必须相等"
    gq = find_primitive_root(q)
    if n.bit_count() == 1:
        # n是一个power of 2，可以正常运行
        
        zeta_n = pow(gq, (q-1)//n, q) # 模q意义下的本原n阶单位根，满足zeta^n=1
        A = ntt_standard(a, zeta_n, n, q)
        B = ntt_standard(b, zeta_n, n, q)
        C = [Ai*Bi%q for Ai,Bi in zip(A,B)]
        c = ntt_standard(C, pow(zeta_n, -1, q), n, q)
        # 乘以n的乘法逆元
        inv_n = pow(n, -1, q)
        c = [ci*inv_n%q for ci in c]
    else:
        N = utils.next_power_of_two(2*n)
        assert (q-1)%N == 0, "convolve_standard要求q-1是一个较大的power of 2"
        aN = np.zeros(shape=(N,), dtype=object)
        aN[:n] = a
        bN = np.zeros(shape=(N,), dtype=object)
        bN[:n] = b
        zeta_N = pow(gq, (q-1)//N, q)
        A = ntt_standard(aN, zeta_N, N, q)
        B = ntt_standard(bN, zeta_N, N, q)
        C = [Ai*Bi%q for Ai,Bi in zip(A,B)]
        cN = ntt_standard(C, pow(zeta_N, -1, q), N, q)
        inv_N = pow(N, -1, q)
        # 手动对折
        c = [(cN[i]+cN[i+n])*inv_N%q for i in range(n)]
    return c


def ntt_rader(f, p, q, inverse=False):
    assert utils.isprime(p)
    assert utils.isprime(q)
    # gp: 质数p的生成元
    gp = find_primitive_root(p)
    # gp的各个次幂
    gp_pows = utils.get_powers(gp, p, p)
    # 找eta: 模q意义下的p阶本原单位根
    eta = find_eta(p, q)
    if inverse:
        eta = pow(eta, -1, q)
    eta_pows = utils.get_powers(eta, p, q)
    # 生成我们要的那俩序列
    n = p-1     # 在本函数中，n代表a,b的长度，也即p-1
    a = [f[gp_pows[(n-v)%n]] for v in range(n)]
    b = [eta_pows[gp_pows[v]] for v in range(n)]
    c = convolve_standard(a, b, q)
    F = np.empty(shape=(p,), dtype=object)
    for u in range(p-1):
        F[gp_pows[u]] = f[0] + c[u]
    F[0] = sum(f)
    if inverse:
        F *= pow(p, -1, q)
    F %= q
    return F