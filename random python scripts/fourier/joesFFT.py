
import numpy as np
from scipy.fft import fft

# \omega_N = e^{2i\pi / N}

def naive_DFT(f):
    root = np.exp(2j*np.pi / len(f))
    k = np.arange(len(f))
    N, K = np.meshgrid(k, k)
    return np.sum(f[N]*root**(N*K), axis=1)

def dec_to_bin(val, digits):
    bin_rep = np.binary_repr(val, digits)
    bin_rep = bin_rep[::-1]
    return int(bin_rep, 2)


def Joe_FFT(f):
    N = len(f)
    digits = int(np.log2(N))
    func = np.vectorize(dec_to_bin)
    indices = func(np.arange(N), digits)
    re_f = f[indices]



f = np.random.uniform(10, size=(8, ))
index = np.arange(8, dtype=np.uint)
digi = int(np.log2(len(index)))
for i in index:
    print(dec_to_bin(i, digi))

new_index = np.vectorize(dec_to_bin)
nwe = new_index(index, digi)

print(f)
print(f[nwe])

