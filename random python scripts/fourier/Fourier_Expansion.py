import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from numba import njit

def fourier_expansion(f, p, N=10):

    max_quad = 10**int(np.log10(N)+2)
    a0 = .5*integrate.quadrature(f, -p, p, maxiter=max_quad)[0]/p
    trig_arg = 2*np.pi/p
    coeffs = np.zeros((N,2))

    for n in range(N):
        abi = \
            np.array([integrate.quadrature(lambda x: f(x)*np.cos((n+1)*x), -p, p, maxiter=max_quad)[0]/p,
                      integrate.quadrature(lambda x: f(x)*np.sin((n+1)*x), -p, p, maxiter=max_quad)[0]/p])
        coeffs[n] = abi
 
    @njit
    def fourier_func(t):
        result = a0*np.ones(t.size, dtype=np.float64)
        n = 1
        for (an, bn) in coeffs:
            result += an*np.cos(trig_arg*n*t) + bn*np.sin(trig_arg*n*t)
            n += 1
        return result

    return fourier_func, coeffs, a0

@njit
def fourier_expansion_exp(c0, cn, time, p):

    result = c0*np.ones(time.size, dtype=np.complex128)
    trig_arg = 2j*np.pi*time/p

    n = 1
    for ci in cn:
        result += ci*np.exp(trig_arg*n)
        n += 1
    return result

@njit
def fourier_expansion_cis(a0, an, bn, time, p):

    assert np.size(an) == np.size(bn), "The arrays for 'a' and 'b' coefficients are of different size."
    trig_arg = 2*np.pi*time/p
    result = .5*a0*np.ones(time.size)

    n = 1
    for ai, bi in zip(an, bn):
        result += ai*np.cos(trig_arg*n) + bi*np.sin(trig_arg*n)
        n += 1
    return result

if __name__ == '__main__':
    func = lambda x: x % np.pi
    period = np.pi

    func_fourier, fourier_coeffs, a0 = fourier_expansion(func, period, N=100)

    time = np.linspace(-5*np.pi, 5*np.pi, 10001)

    print(fourier_coeffs)
    print(a0)


    plt.plot(time, func_fourier(time))
    plt.show()

    plt.plot(range(fourier_coeffs.shape[0]), fourier_coeffs, 'o')
    plt.show()
