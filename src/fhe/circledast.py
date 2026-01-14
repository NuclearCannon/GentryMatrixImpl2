from .key_switch import KeySwitchingKey
import numpy as np
from ..polynomial import ZqiXYW

def _auto_W(poly, n, p):
    assert poly.shape == (2, n, n, p-1)
    tmp = np.zeros(shape=(2, n, n, p-1), dtype=object)
    for i in range(p-1):
        # 移动poly的W^i分量到为W^{-i}
        # 考虑到W^p=1, 可以认为W^{-i}=W^{p-i}
        j = (p-i)%p
        if j == p-1:
            # 这就麻烦了。W^{p-1} = - (W^{p-2}+W^{p-3}+...+W^1+W^0)
            for k in range(p-1):
                tmp[:,:,:,k] -= poly[:,:,:,i]
        else:
            tmp[:,:,:,j] += poly[:,:,:,i]
    return tmp


def create_ksks_for_circledast(sk: ZqiXYW, n, p, q):
    """为circledast_ct准备KSK"""
    # 获取sk的系数形式
    s_coeff = sk.to_coeff()
    assert (s_coeff[:,:,1:,:] == 0).all(), "sk必须与Y无关！"
    s_coeff_T = s_coeff.transpose(0,2,1,3)  # 转置
    s_coeff_T[1] *= -1  # 共轭
    s_coeff_T = _auto_W(s_coeff_T, n, p)    # W->W^{-1}

    sk_conj_T = ZqiXYW(n,p,q,coeff=s_coeff_T)

    # 我们需要准备两个
    # 1.\overline{s}(Y, W^{-1}) -> s
    # 2.s(X,W) * \overline{s}(Y, W^{-1}) -> s
    
    ksk1 = KeySwitchingKey(
        sk_from=sk_conj_T,
        sk_to=sk,
        n=n,p=p,q=q
    )
    ksk2 = KeySwitchingKey(
        sk_from=sk_conj_T * sk,
        sk_to=sk,
        n=n,p=p,q=q
    )
    return ksk1, ksk2


def circledast_ct(ct_u, ct_v, ksks):
    au, bu = ct_u
    av, bv = ct_v
    assert isinstance(au, ZqiXYW)
    assert isinstance(av, ZqiXYW)
    assert isinstance(bu, ZqiXYW)
    assert isinstance(bv, ZqiXYW)
    
    ksk1: KeySwitchingKey = ksks[0]
    ksk2: KeySwitchingKey = ksks[1]

    auav = au.circledast(av) # 此项需要以s(X,W)\overline{s}(Y, W^{-1})为KSK源
    aubv = au.circledast(bv) # 此项权重为s
    buav = bu.circledast(av) # 此项需要以\overline{s}(Y, W^{-1})为KSK源
    bubv = bu.circledast(bv) # 此项权重为1

    buav_ksed = ksk1.key_switch_big(buav)
    auav_ksed = ksk2.key_switch_big(auav)

    a = aubv + buav_ksed[0] + auav_ksed[0]
    b = bubv + buav_ksed[1] + auav_ksed[1]
    return a, b