import sympy
from functools import lru_cache
from . import utils
import math


@lru_cache(maxsize=None)
def _find_zeta_inner(n, q):
    assert (q-1)%(4*n) == 0, f"q={q}, n={n}"
    g = utils.primitive_root(q)
    zeta = pow(g, (q-1)//(4*n), q)
    assert pow(zeta, 2*n, q) == q-1
    return zeta

def find_zeta(n:int, q:int) -> int:
    """找到一个本原单位根zeta使得zeta^{2n}+1=0 (mod q).本函数会自动cache"""
    return _find_zeta_inner(n, q)

@lru_cache(maxsize=None)
def _find_eta_inner(p, q):
    assert (q-1)%p == 0
    g = utils.primitive_root(q)
    eta = pow(g, (q-1)//p, q)
    assert pow(eta, p, q) == 1
    return eta

def find_eta(p:int, q:int) -> int:
    """找到一个本原单位根eta使得eta^{p}=1 (mod q).本函数会自动cache"""
    return _find_eta_inner(p, q)

@lru_cache(maxsize=None)
def _find_prime_q(x: int, start: int = 2) -> int:
    if not isinstance(x, int) or x <= 0:
        raise ValueError("x 必须是正整数")
    if not isinstance(start, int) or start < 2:
        raise ValueError("start 必须是不小于 2 的整数")

    # 特殊情况：x == 1，任意质数都满足条件
    if x == 1:
        result = sympy.nextprime(start - 1, 1)
        assert isinstance(result, int)
        return result

    # 计算最小的 k，使得 q = k*x + 1 >= start
    # 即 k >= (start - 1) / x
    k = (start - 1 + x - 1) // x  # 等价于 ceil((start - 1) / x)
    if k < 1:
        k = 1  # 因为 q = k*x + 1 >= 2 ⇒ k >= 1（当 x >= 1）

    while True:
        q = k * x + 1
        if utils.isprime(q):
            return q
        k += 1


def param_find_q(n: int, p: int, lowerbound: int):
    assert n.bit_count() == 1, "param_find_q: n must be a power-of-2 !"
    assert utils.isprime(p), "param_find_q: p must be a prime!"
    # 找zeta要求4n|(q-1)
    # 找eta要求p|q-1
    # ntt-W要求(p-1)|(q-1)且(p-1)是power of 2
    return _find_prime_q(math.lcm(4*n, p, utils.next_power_of_two(2*(p-1))), lowerbound)

def check_param(n: int, p: int, q:int):
    # n必须为一个power of 2
    assert n.bit_count() == 1, "param_generate: n must be a power-of-2 !"
    # p必须是一个奇数，为了方便我们取它为一个不大的质数，这使得varphi(p)=p-1
    assert utils.isprime(p), "param_generate: p must be a prime!"
    # 模数q应该满足，是质数,q-1能被4n整除，还能被p整除
    # 考虑到p是质数,lcm(4*n, p)==4*n*p
    assert utils.isprime(q), "param_generate: q must be a prime!"
    assert (q%(4*n*p)==1)


__all__ = [
    "find_zeta", "find_eta", "param_find_q", "check_param"
]