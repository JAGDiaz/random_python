import numpy as np
import matplotlib.pyplot as plt


def fourier_expansion_exp(c_0, c_n, en, time, p):
    N, T = np.meshgrid(en, time)
    values = c_n(N) * np.exp((2j * np.pi * N * T) / p)
    values = np.sum(values, axis=1)
    return values + c_0


def fourier_expansion_cis(a_0, a_n, b_n, en, time, p, even=None):
    N, T = np.meshgrid(en, time)
    if even is None:
        values = np.sum(a_n(N)*np.cos((2*np.pi*N*T)/p) + b_n(N)*np.sin((2*np.pi*N*T)/p), axis=1)
    elif even:
        values = np.sum(a_n(N)*np.cos((2*np.pi*N*T)/p), axis=1)
    else:
        values = np.sum(b_n(N)*np.sin((2*np.pi*N*T)/p), axis=1)

    return values + a_0*.5


def abs_t_a_n(en):
    eval_this = lambda x: -4/(np.pi**2 * x**2)
    values = np.zeros(len(en))
    values[::2] = eval_this(en[::2])
    return values


def abs_t_periodic(tea, period):
    tea_copy = ((tea-period*.5) % period)-period*.5
    return np.where(tea_copy < 0, -tea_copy, tea_copy)


abs_t_c_0 = 1
abs_t_period = 2


gibbs_a_0 = .75
gibbs_period = 4.
gibbs_b_n = lambda x: (1/(x*np.pi))*(np.cos(.5*x*np.pi) - (-1)**x)
gibbs_a_n = lambda x: -(1 / (x * np.pi)) * np.sin(.5 * x * np.pi)

sawtooth_a_0 = 0
sawtooth_period = 2*np.pi
sawtooth_a_n = lambda x: 0
sawtooth_b_n = lambda x: 1/x

ex_squared_a_0 = 6.
ex_squared_period = 3.
ex_squared_a_n = lambda x: 9./((np.pi*x)**2)
ex_squared_b_n = lambda x: -9./(np.pi*x)

mystery_a_0 = 1
mystery_period = 2*np.pi
mystery_a_n = lambda x: 1/(3**x)
mystery_b_n = lambda x: 0

para_a_0 = (16/15)*np.pi**4
para_period = 2*np.pi
para_a_n = lambda x: -48*((-1)**x)/(x**4)
para_b_n = lambda x: 0

max_n = 10000
n = np.arange(1, max_n+1).astype(dtype=np.float)
t = np.linspace(-3*np.pi, 3*np.pi, 1001)
tau = np.linspace(-np.pi, np.pi, 1001)


# f = fourier_expansion_cis(sawtooth_a_0, sawtooth_a_n, sawtooth_b_n, n, t, sawtooth_period, even=False)
# f = fourier_expansion_cis(gibbs_a_0, gibbs_a_n, gibbs_b_n, n, t, gibbs_period)
# f = fourier_expansion_cis(ex_squared_a_0, ex_squared_a_n, ex_squared_b_n, n, t, ex_squared_period,
#                          even=None)
f1 = fourier_expansion_cis(mystery_a_0, mystery_a_n, mystery_b_n, n, t, mystery_period, even=True)
# f = abs_t_periodic(t, 2)
f2 = (9-3*np.cos(t))/(10 - 6*np.cos(t))
# f = (t % 3.)**2

# f = fourier_expansion_cis(para_a_0, para_a_n, para_b_n, n, t, para_period, even=True)
# f1 = (np.pi**2 - tau**2)**2

plt.plot(t, f1, '-k')
plt.plot(t, f2, '-b')
#plt.plot(tau, f1, '-b')
plt.grid()
plt.xlabel("x")
plt.ylabel("f(x)")
plt.show()

# print(np.linalg.norm(f-f1, ord=np.inf))
