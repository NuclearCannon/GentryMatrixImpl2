from . import ntt_standard, ntt_W, intt_W
from ...param_gen import find_zeta
from ...utils import get_powers




def ntt_mathcal_I(a, n, q, alter=False):
    zeta = find_zeta(n, q)
    if alter:
        zeta = pow(zeta, -1, q)

    zetas = get_powers(zeta, n, q)  # zeta的0~(n-1)次方
    a2 = [a[i]*zetas[i] % q for i in range(n)]
    omega = zeta**4 % q
    a3 = ntt_standard(a2, omega, n, q)
    return a3
    
    

def intt_mathcal_I(a, n, q, alter=False):
    zeta = find_zeta(n, q)
    if alter:
        zeta = pow(zeta, -1, q)

    inv_zetas = get_powers(pow(zeta, -1, q), n, q)  # zeta的0~(n-1)次方
    # intt_standard使用ntt_standard(zeta^{-1})再除以n来实现
    a2 = ntt_standard(a, pow(zeta, -4, q), n, q)

    inv_n = pow(n, -1, q)
    a3 = [a2[i]*inv_n*inv_zetas[i] % q for i in range(n)]

    return a3


def get_mathcal_I(n,q):
    """ I = zeta^n """
    zeta = find_zeta(n,q)
    I = pow(zeta, n, q)
    assert (I*I+1)%q==0
    return I