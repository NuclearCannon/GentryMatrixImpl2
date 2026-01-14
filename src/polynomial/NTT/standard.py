from functools import lru_cache


def bit_reverse_copy(arr):
    n = len(arr)
    if n <= 1:
        return arr[:]
    bits = n.bit_length() - 1
    if (1 << bits) != n:
        raise ValueError("Length must be a power of two")
    
    rev = _get_bit_reverse_table(n, bits)
    return [arr[rev[i]] for i in range(n)]

@lru_cache(maxsize=None)
def _get_bit_reverse_table(n, bits):
    rev = [0] * n
    for i in range(1, n):
        rev[i] = (rev[i >> 1] >> 1) | ((i & 1) << (bits - 1))
    return tuple(rev)



def ntt_standard(a, root, n, q):
    """Correct iterative Cooley-Tukey DIT NTT"""
    assert pow(root, n, q) == 1
    assert pow(root, n//2, q) == q-1
    original_a = a
    a = a[:]  # make a copy
    # Step 1: bit-reverse the input
    a = bit_reverse_copy(a)
    
    m = 1
    while m < n:
        w_m = pow(root, n // (2 * m), q)  # primitive (2m)-th root
        for i in range(0, n, 2 * m):
            w = 1
            for j in range(i, i + m):
                u = a[j]
                v = (a[j + m] * w) % q
                a[j] = (u + v) % q
                a[j + m] = (u - v) % q
                w = (w * w_m) % q
        m *= 2
    return a  # now in natural order