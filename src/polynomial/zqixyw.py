import numpy as np
from .NTT.ntt_i import ntt_mathcal_I, intt_mathcal_I, ntt_W, intt_W, get_mathcal_I
from ..utils.distribution import generate_uniform

class ZqiXYW:
    """
        多项式环Zq[i][X,Y,W]/(X^n-i, Y^n+i, Phi_p(W))
    """
    def __init__(self, n:int, p:int, q:int|None = None, **kwargs) -> None:
        allowed_keys = {'coeff', 'ntt', 'half', 'zero', 'uniform'}  # 允许的关键字参数
        invalid_keys = set(kwargs) - allowed_keys
        if invalid_keys:
            raise TypeError(f"Ziq3d.__init__() got unexpected keyword argument(s): {', '.join(invalid_keys)}")
    
        self.n = n
        self.p = p
        self.q = q  # 模数
        self.coeff_form: np.ndarray | None = kwargs.get("coeff")    # 系数表示下的实数部分，一个(2,n,n,p-1)的ndarray
        self.ntt_form: np.ndarray | None = kwargs.get("ntt")    # 全NTT表示。我们会把NTT结果暂存在这里
        # i轴的NTT会把(a+bi)分解为两个整数(a+bI, a-bI)
        # 其中，I是一个Zq中的特殊整数使得I^2+1==0(mod q)
        # 我们记录P=a+bI, N=a-bI
        self.half_form: np.ndarray | None = kwargs.get("half")    # i,W轴是NTT表示的，X,Y轴是系数表示的

        if kwargs.get("zero", False):
            self.coeff_form = np.zeros((2,n,n,p-1), dtype=object)
            self.ntt_form = np.zeros((2,n,n,p-1), dtype=object)
            self.half_form = np.zeros((2,n,n,p-1), dtype=object)
        elif kwargs.get("uniform", False):
            assert self.q is not None
            # 通常来讲，uniform生成的多项式是用来乘的，因此我们生成ntt形式的均匀随机数
            self.ntt_form = generate_uniform((2,n,n,p-1), self.q)

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
                P[i,j,:] = ntt_W(P[i,j,:], p, q)
                N[i,j,:] = ntt_W(N[i,j,:], p, q)
        return np.stack([P,N], axis=0)
    
    @staticmethod
    def _iw_intt(half_form, n, p, q):
        I = get_mathcal_I(n,q)
        P = half_form[0]
        N = half_form[1]
        # P: (X^n-I, Y^n+I, Phi_p(W))
        # N: (X^n+I, Y^n-I, Phi_p(W))
        for i in range(n):
            for j in range(n):
                P[i,j,:] = intt_W(P[i,j,:], p, q)
                N[i,j,:] = intt_W(N[i,j,:], p, q)
        real = (P+N)*pow(2,-1,q)%q
        imag = (P-N)*pow(2*I,-1,q)%q
        return np.stack([real,imag], axis=0)
    
    @staticmethod
    def _xy_ntt(half_form, n, p, q):
        arr = half_form.copy()
        for i in range(n):
            for j in range(p-1):
                arr[0,i,:,j] = ntt_mathcal_I(arr[0,i,:,j], n, q, True)
                arr[1,i,:,j] = ntt_mathcal_I(arr[1,i,:,j], n, q, False)
        for i in range(n):
            for j in range(p-1):
                arr[0,:,i,j] = ntt_mathcal_I(arr[0,:,i,j], n, q, False)
                arr[1,:,i,j] = ntt_mathcal_I(arr[1,:,i,j], n, q, True)
        return arr
    
    @staticmethod
    def _xy_intt(ntt_form, n, p, q):
        arr = ntt_form.copy()
        for i in range(n):
            for j in range(p-1):
                arr[0,i,:,j] = intt_mathcal_I(arr[0,i,:,j], n, q, True)
                arr[1,i,:,j] = intt_mathcal_I(arr[1,i,:,j], n, q, False)
        for i in range(n):
            for j in range(p-1):
                arr[0,:,i,j] = intt_mathcal_I(arr[0,:,i,j], n, q, False)
                arr[1,:,i,j] = intt_mathcal_I(arr[1,:,i,j], n, q, True)
        return arr


    @classmethod
    def from_real_imag(cls, real:np.ndarray, imag: np.ndarray, n:int, p:int, q:int|None = None):
        """生成一个多项式，它的实部和虚部分别由一个(shape=(n,n,p-1),dtype=object)的ndarray给出"""
        # 检查输入
        assert isinstance(real, np.ndarray)
        assert real.shape == (n,n,p-1)
        assert real.dtype == object
        assert isinstance(imag, np.ndarray)
        assert imag.shape == (n,n,p-1)
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
        if self.ntt_form is not None:
            kwargs["ntt"] = - self.ntt_form
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
            print("Warning: __add__中两个多项式没有共同表示形式，我们不得不确保去转化为系数形式")
            self._ensure_coeff_form_exists()
            other._ensure_coeff_form_exists()
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


        

    def circledast(self, other: "ZqiXYW"):
        assert self.like(other)
        assert self.q is not None
        # half形式是有利于circledast的
        self._ensure_half_form_exists()
        other._ensure_half_form_exists()
        n,p,q = self.npq()
        result = np.zeros((2,n,n,p-1))    # 结果的half形式
        A = self.half_form
        B = self.half_form
        assert isinstance(A, np.ndarray)
        assert isinstance(B, np.ndarray)
        assert A.shape == B.shape == (2,n,n,p-1)

        for i in range(p-1):
            # 对self的第i个W分量和other的第p-2-i个W分量进行2d circledast，结果放在result[:,:,:,i]中
            a = A[:,:,:,i]
            b = B[:,:,:,p-2-i]
            # 提取P,N分量
            ap, an = a[0], a[1]
            bp, bn = b[1].T, b[0].T         # 在这里对B共轭转置
            # 做矩阵乘法
            result[0,:,:,i] = ap @ bp % q
            result[1,:,:,i] = an @ bn % q

        return ZqiXYW(n,p,q,half=result)
    


    




    



    