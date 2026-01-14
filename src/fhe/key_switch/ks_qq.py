from . import encrypt_XYW, param_gen, decrypt_XYW
from ...polynomial import ZqiXYW

import numpy as np
import math


class KeySwitchingKey:
    """更为保守的KSK，模Q而不是q*qo"""
    def __init__(
            self, 
            sk_from: ZqiXYW, sk_to: ZqiXYW, 
            n, p, q,
            B:int = 16,
            no_error_ksk = False,
            no_a_part_ksk = False
                 
                 
    ):
        """生成KSK(sk_from->sk_to)"""
        self.n = n
        self.p = p
        self.q = q
        # 实际上，加密qo * sk_from * B^k 为密文，以q*qo为模数
        # 我们把KS过程放到一个更大的模数Q上去进行，而不是q
        # 我们选择的参数Q,B,L需要满足：
        # 1.原来的a(mod q)能被(B,L)分解，也就是说，B^L>q
        # 2.Q同样是满足(n,p)要求的大质数
        # 3.为了确保降模构成的噪声压缩程度足够大，Q/q>B
        # Q>n*B*q，2*n*B是对\sum{a_i e_i}的合理估计
        # 留给后人：注意：选取更大的Q将会明显减小KS过程造成的噪声
        Q = param_gen.param_find_q(n, p, 2*n*B*q+1)
        L = int(math.log(q, B))+1

        self.Q = Q
        self.B = B
        self.L = L

        self.cts = []
        sk_from_raw = sk_from.to_coeff() % q
        sk_from_raw[sk_from_raw>(q//2)] -= q
        sk_to_Q = sk_to.switch_q(Q)

        for i in range(L):
            m = (B**i) * sk_from_raw
            if not (abs(m)<q).all():
                print("Warning: KSK生成: sk_from越过了q")
            m %= q
            m = (m * Q) // q
            m %= Q
            m2 = ZqiXYW(n,p,Q,coeff=m)   # 转为多项式对象
            sk_a, sk_b = encrypt_XYW(m2, sk_to_Q, no_error=no_error_ksk, no_a_part=no_a_part_ksk)
            self.cts.append((sk_a, sk_b))



    def key_switch_big(self, a:ZqiXYW, b:ZqiXYW|None = None):
        # 准备参数
        n = self.n
        p = self.p
        q = self.q
        Q = self.Q


        # 把a按B切片
        a_list = []
        a_coeff = a.to_coeff() % q

        while (a_coeff!=0).any():
            piece = a_coeff % self.B
            a_coeff //= self.B
            # 把这一片视为更大的多项式对象
            a_list.append(ZqiXYW(n,p,Q,coeff=piece))
            

        assert len(a_list) <= self.L, f"KSK长度不足！至少需要{len(a_list)}来实行KS,但是实际上只有{self.L}"
        # a_list[i]是原文在B^i上的分量，将它乘以我们的B^i倍数sk的密文（多项式乘法）
        sum_a_ntt = ZqiXYW(n,p,Q,zero=True)
        sum_b_ntt = ZqiXYW(n,p,Q,zero=True)

        for i,piece in enumerate(a_list):
            sk_a2, sk_b2 = self.cts[i]
            sum_a_ntt += piece * sk_a2
            sum_b_ntt += piece * sk_b2

        sum_a = sum_a_ntt.to_coeff()
        sum_b = sum_b_ntt.to_coeff()

        sum_a %= Q
        sum_b %= Q

        sum_a = (sum_a * q + (Q//2)) // Q
        sum_b = (sum_b * q + (Q//2)) // Q

        sum_a = ZqiXYW(n,p,q,coeff=sum_a)
        sum_b = ZqiXYW(n,p,q,coeff=sum_b)
        # 补上一开始的常数项
        if b is not None:
            sum_b += b
        return sum_a, sum_b
