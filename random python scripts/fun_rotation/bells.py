#! /usr/bin/python3

import numpy as np

def bell(n):
	if n== 0 or n == 1:
		return 1
	else:
		return sum([stirling2(n,k) for k in range(n+1)])

def stirling2(n,k):
	if n == k or k == 1:
		return 1
	elif k == 0:
		return 0
	else:
		return k*stirling2(n-1,k) + stirling2(n-1,k-1)

def tailStirl(n, k1, k2, a, b):
    if n == k1 or k2 == 1:
        return a
    elif k2 == 0:
        return b
    else:
        return tailStirl(n-1, k1, k1-1, b, k2*b + b)

def tailAdd(n, a, b):
	if n == 0:
		return a
	if n == 1:
		return b
	return tailAdd(n - 1, b, a + b)

def fib(n):
	return tailAdd(n, 0, 1)

def lucas(n):
	return tailAdd(n, 2, 1)

def tailnCk(n, k1, k2, a, b):
    if n == k2 or k2 == 0:
        return a

x = range(1,11)
for i in x:
    print(stirling2(len(x),i))
