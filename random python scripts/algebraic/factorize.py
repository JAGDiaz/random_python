#! /usr/bin/python3

import numpy as np
import sys

def factors(n, as_array = False):
	if(n < 0):
		return "-%s" % factors(-n)
	if(n < 4):
		return n
	d = np.arange(2, int(np.sqrt(n))+1)
	f = n % d
	indices = np.where(f == 0)[0]
	if(as_array):
		if(len(indices) == 0):
			return np.array([n])
		return np.append([d[indices[0]]], factors(n//d[indices[0]], as_array))
	else:	
		if(len(indices) == 0):
			return "%d" % n
		return "%d * %s" % (d[indices[0]], factors(n//d[indices[0]]))

products = [int(arg) for arg in sys.argv[1:]]

for p in products:
	print("%9d = %s" % (p, factors(p)))
