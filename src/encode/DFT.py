# 本文件实现DFT
import numpy as np
import random
from ..param_gen import find_primitive_root

def naive_dft_XY_complex(arr, n, using_conj_zetas=False):
    assert len(arr) == n
    results = []
    if using_conj_zetas:
        zeta = np.exp(-1j * np.pi / (2*n))   # exp(-i*pi/2n)，满足zeta^n=-i
    else:
        zeta = np.exp(1j * np.pi / (2*n))   # exp(i*pi/2n)，满足zeta^n=i
    for k in range(n):
        Xk = 0
        for j in range(n):
            Xk += arr[j] * np.power(zeta, pow(5,k,4*n)*j)
        results.append(Xk)
    return results



def naive_idft_XY_complex(arr, n, using_conj_zetas=False):
    assert len(arr) == n
    results = []
    if using_conj_zetas:
        zeta = np.exp(-1j * np.pi / (2*n))   # exp(-i*pi/2n)，满足zeta^n=-i
    else:
        zeta = np.exp(1j * np.pi / (2*n))   # exp(i*pi/2n)，满足zeta^n=i
    for j in range(n):
        Xj = 0
        for k in range(n):
            Xj += arr[k] * np.power(zeta, -pow(5,k,4*n)*j)
        Xj /= n   # 除以n
        results.append(Xj)
    return results

def naive_dft_W_complex(arr, p):
    assert len(arr) == p-1
    gamma = find_primitive_root(p)
    results = []
    eta = np.exp(2j * np.pi / p)    # 满足eta^p=1
    for k in range(1,p):
        Xk = 0
        for j in range(p-1):
            Xk += arr[j] * np.power(eta, j*pow(gamma, k, p))
        results.append(Xk)
    return results


def naive_idft_W_complex(arr, p):
    assert len(arr) == p-1
    gamma = find_primitive_root(p)
    tmps = []
    eta = np.exp(2j * np.pi / p)
    for j in range(p-1):
        tmpj = 0
        for k in range(1,p):
            tmpj += arr[k-1] * np.power(eta, -j*pow(gamma, k, p))
        tmps.append(tmpj)
    A = sum(tmps)
    xs = [(tmps[i] + A)/p for i in range(p-1)]
    # print("arr", arr)
    # print("xs", xs)
    # input()
    return xs

