import numpy as np


def swap_rows(l, i1, i2):
    l[i1], l[i2] = l[i2], l[i1]

def swap_cols(l, j1, j2):
    for r in l:
        r[j1], r[j2] = r[j2], r[j1]

def print_list(l):
    print(f"[{l[0]},")
    for r in l[1:-1]:
        print(f" {r},")
    print(f" {l[-1]}]")

def optimize_minor(array):
    n = len(array)
    assert not n % 2, "Blah blah blah."
    n //= 2

    minor_pos = [(i,j) for i in range(n) for j in range(n)]

    

l = [[10*j + i for i in range(10)] for j in range(10)]

optimize_minor(l)