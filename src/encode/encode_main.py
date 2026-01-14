import numpy as np
from . import DFT
from ..polynomial import ZqiXYW



class Encoder:
    def __init__(self, n, p, q) -> None:
        self.n = n
        self.p = p
        self.q = q

    def _encode_2d(self, arr, delta):
        n = self.n
        assert arr.shape == (n,n)
        # 先DFT，再放缩
        a = arr.copy()
        for i in range(n):
            a[i, :] = DFT.naive_idft_XY_complex(a[i, :], n, using_conj_zetas=True)
        for i in range(n):
            a[:, i] = DFT.naive_idft_XY_complex(a[:, i], n, using_conj_zetas=False)
        # 分虚实部
        real = np.zeros(shape=(n, n), dtype=object)
        imag = np.zeros(shape=(n, n), dtype=object)
        for i in range(n):
            for j in range(n):
                real[i,j] += round(a[i,j].real * delta)
                imag[i,j] += round(a[i,j].imag * delta)

        return real, imag



    def _decode_2d(self, real, imag):
        n, q = self.n, self.q
        a = np.zeros((n, n), dtype=object)

        real = real % q
        real[real > (q//2)] -= q
        imag = imag % q
        imag[imag > (q//2)] -= q
        a = real + 1j * imag

        for i in range(n):
            a[i, :] = DFT.naive_dft_XY_complex(a[i, :], n, using_conj_zetas=True)
        for i in range(n):
            a[:, i] = DFT.naive_dft_XY_complex(a[:, i], n, using_conj_zetas=False)
        return a

    def _encode_3d(self, arr, delta):
        n, p = self.n, self.p
        assert arr.shape == (n,n,p-1)
        result_r = np.zeros(shape=(n, n, p-1), dtype=object)
        result_i = np.zeros(shape=(n, n, p-1), dtype=object)

        tmp = np.zeros(shape=(n,n,p-1), dtype=complex)
        # 先做W-iDFT
        for i in range(n):
            for j in range(n):
                tmp[i,j,:] = DFT.naive_idft_W_complex(arr[i,j,:], p)

        # 在对每个W分量做
        for k in range(p-1):
            r,i = self._encode_2d(tmp[:,:,k], delta)
            result_r[:,:,k] = r
            result_i[:,:,k] = i
        return result_r, result_i

    def _decode_3d(self, real, imag):
        n, p = self.n, self.p
        tmp = np.zeros(shape=(n,n,p-1), dtype=complex)
        result = np.zeros(shape=(n,n,p-1), dtype=complex)

        for k in range(p-1):
            tmp[:,:,k] = self._decode_2d(real[:,:,k], imag[:,:,k])
        for i in range(n):
            for j in range(n):
                result[i,j,:] = DFT.naive_dft_W_complex(tmp[i,j,:], p)
        return result
    

    def encode_to_polynomial(self, arr, delta) -> ZqiXYW:
        r, i = self._encode_3d(arr, delta)
        return ZqiXYW.from_real_imag(r,i, self.n, self.p, self.q)
    
    def decode_from_polynomial(self, poly: ZqiXYW):
        r, i = poly.to_real_imag()
        return self._decode_3d(r, i)

        
        



