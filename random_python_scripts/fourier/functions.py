
import numpy as np
from scipy.fft import fft
import matplotlib.pyplot as plt

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

def discrete_conv(x,y):
    z = np.zeros(shape=x.size+y.size-1, dtype=np.float64)

    for ii in range(x.size):
        for jj in range(y.size):
            z[ii+jj] += x[ii]*y[jj]

    return z

def discrete_conv_alt(x, y):
    x_size, y_size = x.size, y.size
    y_here = np.concatenate((np.zeros(x_size-1), y, np.zeros(x_size-1)))
    x_here = np.copy(x[::-1])
    z = np.zeros(x_size + y_size - 1)

    for ii in range(z.size):
        z[ii] = np.dot(x_here, y_here[ii:ii+x_size])

    return z

if __name__ == "__main__":

    x = np.linspace(-2.5, 2.5, 501)
    y1 = np.ones(x.size)
    y2 = -x

    new_signal = discrete_conv_alt(y1, y2)

    fig, (ax1,ax2,ax3) = plt.subplots(3)

    ax1.plot(y1, 'k-'); ax1.set_ylabel("$y_1$")
    ax2.plot(y2, 'k-'); ax2.set_ylabel("$y_2$")
    ax3.plot(new_signal, 'k-'); ax3.set_ylabel("$y_1 * y_2$")
    ax3.set_xlabel("Sample #")

    plt.show()
    plt.close('all')

    # a = np.array([[(ii + jj, ii, jj) for jj in range(10)] for ii in range(10)], dtype=tuple)

    # print(a)


    "ZEROES + ONES = BINARY"