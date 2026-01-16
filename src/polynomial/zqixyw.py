import numpy as np
from .NTT.ntt_i import ntt_mathcal_I, intt_mathcal_I, ntt_W, intt_W, get_mathcal_I
from ..utils.distribution import generate_uniform
from ..utils import logger

class ZqiXYW:
    """
        多项式环Zq[i][X,Y,W]/(X^n-i, Y^n+i, Phi_p(W))
        底层是一个shape=(2,p-1,n,n)的数组
    """
    def __init__(self, n:int, p:int, q:int, **kwargs) -> None:
        assert isinstance(n,int)
        assert isinstance(p,int)
        assert isinstance(q,int)
        allowed_keys = {'coeff', 'ntt', 'half', 'zero', 'uniform'}  # 允许的关键字参数
        invalid_keys = set(kwargs) - allowed_keys
        if invalid_keys:
            raise TypeError(f"Ziq3d.__init__() got unexpected keyword argument(s): {', '.join(invalid_keys)}")
    
        self.n = n
        self.p = p
        self.q = q  # 模数
        self.coeff_form: np.ndarray | None = kwargs.get("coeff")    # 系数表示下的实数部分，一个(2,p-1,n,n)的ndarray
        self.ntt_form: np.ndarray | None = kwargs.get("ntt")    # 全NTT表示。我们会把NTT结果暂存在这里
        # i轴的NTT会把(a+bi)分解为两个整数(a+bI, a-bI)
        # 其中，I是一个Zq中的特殊整数使得I^2+1==0(mod q)
        # 我们记录P=a+bI, N=a-bI
        self.half_form: np.ndarray | None = kwargs.get("half")    # i,W轴是NTT表示的，X,Y轴是系数表示的

        if kwargs.get("zero", False):
            self.coeff_form = np.zeros((2,p-1,n,n), dtype=object)
            self.ntt_form = np.zeros((2,p-1,n,n), dtype=object)
            self.half_form = np.zeros((2,p-1,n,n), dtype=object)
        elif kwargs.get("uniform", False):
            assert self.q is not None
            # 通常来讲，uniform生成的多项式是用来乘的，因此我们生成ntt形式的均匀随机数
            self.ntt_form = generate_uniform((2,p-1,n,n), self.q)

    @staticmethod
    def _iw_ntt(coeff_form, n, p, q):
        I = get_mathcal_I(n,q)
        real, imag = coeff_form[0], coeff_form[1]
        P = real + imag*I
        N = real - imag*I
        # P: (X^n-I, Y^n+I, Phi_p(W))，即i=I的情况
        # N: (X^n+I, Y^n-I, Phi_p(W))
        for i in range(n):
            for j in range(n):
                P[:,i,j] = ntt_W(P[:,i,j], p, q)
                N[:,i,j] = ntt_W(N[:,i,j], p, q)
        result = np.stack([P,N], axis=0) % q
        return result
    
    @staticmethod
    def _iw_intt(half_form, n, p, q):
        I = get_mathcal_I(n,q)
        P = half_form[0]
        N = half_form[1]
        P2 = np.zeros_like(P)
        N2 = np.zeros_like(N)
        # P: (X^n-I, Y^n+I, Phi_p(W))
        # N: (X^n+I, Y^n-I, Phi_p(W))
        for i in range(n):
            for j in range(n):
                P2[:,i,j] = intt_W(P[:,i,j], p, q)
                N2[:,i,j] = intt_W(N[:,i,j], p, q)
        real = (P2+N2)*pow(2,-1,q)%q
        imag = (P2-N2)*pow(2*I,-1,q)%q
        result = np.stack([real,imag], axis=0) % q
        return result
    
    @staticmethod
    def _xy_ntt(half_form, n, p, q):
        arr = half_form.copy()
        for w in range(p-1):
            # 对Y轴进行NTT
            for x in range(n):
                arr[0,w,x,:] = ntt_mathcal_I(arr[0,w,x,:], n, q, True)
                arr[1,w,x,:] = ntt_mathcal_I(arr[1,w,x,:], n, q, False)
            # 对X轴进行NTT
            for y in range(n):
                arr[0,w,:,y] = ntt_mathcal_I(arr[0,w,:,y], n, q, False)
                arr[1,w,:,y] = ntt_mathcal_I(arr[1,w,:,y], n, q, True)
        return arr
    
    @staticmethod
    def _xy_intt(ntt_form, n, p, q):
        arr = ntt_form.copy()
        for w in range(p-1):
            # 对Y轴进行iNTT
            for x in range(n):
                arr[0,w,x,:] = intt_mathcal_I(arr[0,w,x,:], n, q, True)
                arr[1,w,x,:] = intt_mathcal_I(arr[1,w,x,:], n, q, False)
            # 对X轴进行iNTT
            for y in range(n):
                arr[0,w,:,y] = intt_mathcal_I(arr[0,w,:,y], n, q, False)
                arr[1,w,:,y] = intt_mathcal_I(arr[1,w,:,y], n, q, True)
        return arr


    @classmethod
    def from_real_imag(cls, real:np.ndarray, imag: np.ndarray, n:int, p:int, q:int):
        """生成一个多项式，它的实部和虚部分别由一个(shape=(n,n,p-1),dtype=object)的ndarray给出"""
        # 检查输入
        assert isinstance(real, np.ndarray)
        assert real.shape == (p-1,n,n)
        assert real.dtype == object
        assert isinstance(imag, np.ndarray)
        assert imag.shape == (p-1,n,n)
        assert imag.dtype == object
        result = ZqiXYW(n,p,q)
        result.coeff_form = np.stack([real,imag], axis=0)
        return result

    

    def to_real_imag(self) -> "tuple[np.ndarray, np.ndarray]":
        self._ensure_coeff_form_exists()
        assert self.coeff_form is not None
        return self.coeff_form[0], self.coeff_form[1]
    
    def to_coeff(self) -> "np.ndarray":
        self._ensure_coeff_form_exists()
        assert self.coeff_form is not None
        return self.coeff_form.copy()

    def npq(self):
        return self.n, self.p, self.q


    def like(self, other: "ZqiXYW") -> bool:
        assert isinstance(other, ZqiXYW)
        return self.npq() == other.npq()
    
    def switch_q(self, Q:int):
        """生成一个新的多项式，它和自己具有相同的系数表示（？），但是是另一个模数。这也可以用于创建sk的有模数形式"""
        assert isinstance(Q, int)
        coeff = self.to_coeff()
        assert isinstance(coeff, np.ndarray)
        if self.q is not None:
            coeff %= self.q
            coeff[coeff > (self.q//2)] -= self.q
            if np.max(np.abs(coeff)) > self.q//8:
                print("switch_q: 危险！")
        return ZqiXYW(self.n, self.p, Q, coeff=coeff)
    
    def __neg__(self):
        n,p,q = self.npq()
        kwargs = {}
        if self.coeff_form is not None:
            kwargs["coeff"] = - self.coeff_form
        if self.half_form is not None:
            kwargs["half"] = - self.half_form
        elif self.ntt_form is not None:
            kwargs["ntt"] = - self.ntt_form
        else:
            raise ValueError("__neg__: self是三无多项式")
        return ZqiXYW(n,p,q,**kwargs)
        
    def __sub__(self, other):
        return self + (-other)


    def __add__(self, other: "ZqiXYW"):
        assert self.like(other)
        result = ZqiXYW(self.n, self.p, self.q)
        valid = False
        if self.coeff_form is not None and other.coeff_form is not None:
            result.coeff_form = self.coeff_form + other.coeff_form
            valid = True
        if self.half_form is not None and other.half_form is not None:
            result.half_form = self.half_form + other.half_form
            valid = True
        if self.ntt_form is not None and other.ntt_form is not None:
            result.ntt_form = self.ntt_form + other.ntt_form
            valid = True
        if valid:
            return result
        else:
            logger.log("Warning: __add__中两个多项式没有共同表示形式，我们不得不确保转化为NTT形式")
            # NTT是最通用的形式，让我们在大部分情况下使用纯NTT表示
            self._ensure_ntt_form_exists()
            other._ensure_ntt_form_exists()
            return self + other # 重新执行一次
        
    
    def _ensure_coeff_form_exists(self):
        """确保系数状态存在，如果不存在，算一个出来"""
        assert self.q is not None
        if self.coeff_form is None:
            self._ensure_half_form_exists()
            self.coeff_form = ZqiXYW._iw_intt(self.half_form, self.n, self.p, self.q)

            

    def _ensure_half_form_exists(self):
        """确保中间表示状态存在，如果不存在，算一个出来"""
        assert self.q is not None
        if self.half_form is None:
            if self.coeff_form is not None:
                self.half_form = ZqiXYW._iw_ntt(self.coeff_form, self.n, self.p, self.q)
            elif self.ntt_form is not None:
                self.half_form = ZqiXYW._xy_intt(self.ntt_form, self.n, self.p, self.q)
            else:
                raise ValueError("你的多项式三种状态怎么都是None")

    def _ensure_ntt_form_exists(self):
        """确保NTT状态存在，如果不存在，算一个出来"""
        assert self.q is not None
        if self.ntt_form is None:
            self._ensure_half_form_exists()
            self.ntt_form = ZqiXYW._xy_ntt(self.half_form, self.n, self.p, self.q)


    def __mul__(self, other: "ZqiXYW"):
        assert self.like(other)
        # 乘法需要NTT，因此必须有模数
        assert self.q is not None
        self._ensure_ntt_form_exists()
        other._ensure_ntt_form_exists()
        assert isinstance(self.ntt_form, np.ndarray)
        assert isinstance(other.ntt_form, np.ndarray)

        result_ntt = self.ntt_form * other.ntt_form % self.q
        return ZqiXYW(self.n, self.p, self.q, ntt=result_ntt)


    def circledast(self, other: "ZqiXYW", hint="half", force=None):
        # N+N -> N
        # H+H -> H
        # 都有/都没有->hint
        if force == "ntt":
            return self.circledast_on_ntt_form(other)
        if force == "half":
            return self.circledast_on_half_form(other)
        
        N1 = self.ntt_form is not None
        N2 = other.ntt_form is not None
        H1 = self.half_form is not None
        H2 = other.half_form is not None
        N = N1 and N2
        H = H1 and H2
        if N and not H:
            return self.circledast_on_ntt_form(other)
        elif H and not N:
            return self.circledast_on_half_form(other)
        elif hint == "half":
            return self.circledast_on_half_form(other)
        else:
            return self.circledast_on_ntt_form(other)


    def circledast_on_half_form(self, other: "ZqiXYW"):
        assert self.like(other)
        assert self.q is not None
        # half形式是有利于circledast的
        self._ensure_half_form_exists()
        other._ensure_half_form_exists()
        n,p,q = self.npq()
        result = np.zeros((2,p-1,n,n), dtype=object)    # 结果的half形式
        A = self.half_form
        B = other.half_form
        assert isinstance(A, np.ndarray)
        assert isinstance(B, np.ndarray)
        assert A.shape == B.shape == (2,p-1,n,n)

        for w in range(p-1):
            # 对self的第i个W分量和other的第p-2-i个W分量进行2d circledast，结果放在result[:,:,:,i]中
            a = A[:,w,:,:]
            b = B[:,p-2-w,:,:]
            # 提取P,N分量
            ap, an = a[0], a[1]
            bp, bn = b[1].T, b[0].T         # 在这里对B共轭转置
            # 做矩阵乘法
            result[0,w,:,:] = ap @ bp % q   # (+,-) (+,-) => (+,-)
            result[1,w,:,:] = an @ bn % q   # (-,+) (-,+) => (-,+)

        result %= q

        result = ZqiXYW(n,p,q,half=result)
        return result
    
    def circledast_on_ntt_form(self, other: "ZqiXYW"):
        """一种可以直接在NTT形式上进行Circledast运算的方法"""
        assert self.like(other)
        self._ensure_ntt_form_exists()
        other._ensure_ntt_form_exists()
        n,p,q = self.npq()
        result = np.zeros((2,p-1,n,n), dtype=object)    # 结果的half形式
        A = self.ntt_form
        B = other.ntt_form
        assert isinstance(A, np.ndarray)
        assert isinstance(B, np.ndarray)
        assert A.shape == B.shape == (2,p-1,n,n)

        for w in range(p-1):
            # 对self的第i个W分量和other的第p-2-i个W分量进行2d circledast，结果放在result[:,:,:,i]中
            a = A[:,w,:,:]
            b = B[:,p-2-w,:,:]   # B: Y->Y^{-1}
            # 提取P,N分量
            ap, an = a[0], a[1]
            bp, bn = b[1].T, b[0].T         # 在这里对B共轭转置
            # 做矩阵乘法
            result[0,w,:,:] = ap @ bp % q
            result[1,w,:,:] = an @ bn % q

        result *= pow(n,-1,q)
        result %= q

        result = ZqiXYW(n,p,q,ntt=result)
        return result

    
    def __eq__(self, other: "ZqiXYW"):
        assert self.like(other)
        error = self.to_coeff() - other.to_coeff()
        if self.q:
            error %= self.q
        return (error == 0).all()
        


    




    



    