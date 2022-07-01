import numpy as np



def rhs(t, y):
    return t*y


def y_actual(t):
    return np.sqrt(np.exp(t**2))


def eulers_method(t0, y0, h, t_final, f=rhs):
    ts = np.arange(t0, t_final, step=h)
    ys = np.zeros(ts.size)
    ys[0] = y0
    for i, t in enumerate(ts[:-1]):
        ys[i+1] = ys[i] + h*f(t, ys[i])
    return ts, ys


y_init = 1
t_init = 0
dh = 1e-6
t_f = 4

t_approx, y_approx = eulers_method(t_init, y_init, dh, t_f)
print(y_approx[-1])
print(y_actual(t_approx[-1]))

