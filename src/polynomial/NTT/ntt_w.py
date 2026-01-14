from . import utils
from . import find_eta, ntt_rader
import numpy as np


# 高于此阈值时，会使用rader NTT来加速ntt_W，否则朴素地ntt_W
NTT_W_THRESHOLD = 170


def ntt_W_naive(arr, p, q):
    assert utils.isprime(q), "NTT使用的模数必须是质数"
    eta = find_eta(p, q)
    assert len(arr) == p-1
    eta_powers = utils.get_powers(eta, p, q)    # 获取eta^i
    # 注意，eta^{p}==1，其指数可以模p-1
    results = np.empty(shape=(p-1,),dtype=object)
    for k in range(1,p):
        Xk = 0
        for j in range(p-1):
            # Xk += arr[j] * pow(eta, k * j, q)
            Xk += arr[j] * eta_powers[(k*j)%p]
        Xk %= q
        results[k-1] = Xk
    return results

def ntt_W(arr, p, q):
    """
        这里，模多项式是Phi_p(X)=1+X+X^2+...+X^{p-2}
        p是一个质数
        q是模数（也是质数）
        由质数的性质还有，eta^{p-1}=1
    """
    if p > NTT_W_THRESHOLD:
        return ntt_W_rader(arr, p, q)
    else:
        return ntt_W_naive(arr, p, q)



def intt_W_naive(arr, p, q):
    assert len(arr) == p-1
    assert utils.isprime(q), "NTT使用的模数必须是质数"
    # 你会发现本文件实现的和一般意义上的NTT有所不同，它引入了一个称为tmps的临时序列
    eta = find_eta(p, q)
    eta_powers = utils.get_powers(pow(eta,-1,q), p, q)    # 获取eta^{-i}
    tmps = np.empty(shape=(p-1,),dtype=object)
    for j in range(p-1):
        tmpj = 0
        for k in range(1,p):
            # tmpj += arr[k-1] * pow(eta, -k*j, q)
            tmpj += arr[k-1] * eta_powers[(k*j)%p]
        tmpj %= q
        tmps[j] = tmpj
    A = sum(tmps)
    p_inv = pow(p,-1,q)
    result = (tmps + A) * p_inv % q
    return result

def intt_W(arr, p, q):
    if p > NTT_W_THRESHOLD:
        return intt_W_rader(arr, p, q)
    else:
        return intt_W_naive(arr, p, q)



def ntt_W_rader(arr, p:int, q):
    a = list(arr) + [0]
    tmp = ntt_rader(a, p, q, inverse=False)
    return tmp[1:]

def intt_W_rader(arr, p, q):
    a = [0] + list(arr) # 假装F[0] == 0
    tmp = ntt_rader(a, p, q, inverse=True)
    F0 = -tmp[p-1]
    tmp += F0
    return tmp[:p-1] % q
