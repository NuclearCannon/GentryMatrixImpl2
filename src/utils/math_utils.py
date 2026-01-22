import sympy
from functools import lru_cache

@lru_cache(maxsize=None)
def _isprime_inner(x: int) -> bool:
    return sympy.isprime(x)


def isprime(x: int) -> bool:
    """判断质数。本函数会自动cache结果"""
    return _isprime_inner(x)


def next_power_of_two(x: int) -> int:
    """获取一个尽可能小的y使得y是power-of-2且y>=x"""
    if x <= 1:
        return 1
    # 找到大于等于 target 的最小 2 的幂
    y = 1
    while y < x:
        y <<= 1
    return y


@lru_cache(maxsize=None)
def get_powers(zeta:int, N:int, q:int) -> tuple[int, ...]:
    return tuple(pow(zeta, j, q) for j in range(N))


@lru_cache(maxsize=None)
def _primitive_root_inner(x: int) -> int:
    r = sympy.primitive_root(x)
    assert isinstance(r, int)
    return r

def primitive_root(q):
    return _primitive_root_inner(q)