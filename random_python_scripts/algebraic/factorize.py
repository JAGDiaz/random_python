#! /usr/bin/python3

import numpy as np
import sys
from numba import njit

def factors_bad(n, as_array=True):
	if n < 4:
		return n
	d = np.arange(2, int(np.sqrt(n))+1)
	f = n % d
	indices = np.where(f == 0)[0]
	if as_array :
		if len(indices) == 0:
			return np.array([n])
		return np.append([d[indices[0]]], factors_bad(n//d[indices[0]], as_array))
	else:	
		if(len(indices) == 0):
			return f"{n}"
		return f"{d[indices[0]]} * {factors_bad(n//d[indices[0]])}"


def factors(k):

	if k < 4:
		yield k
		return

	last = 2
	while True:
		ii = last

		while k % ii:
			ii += 1

		yield ii

		if ii > last:
			last = ii
		k //= ii

		if k == 1:
			return

def list_to_str_product(the_list):
	
	if len(the_list) == 0:
		return "[]"

	if len(the_list) == 1:
		return f"{the_list[0]}"

	the_str = f"{the_list[0]} *"
	for item in the_list[1:-1]:
		the_str += f" {item} *"
	the_str += f" {the_list[-1]}"

	return the_str



if __name__ == "__main__":

	products = [int(arg) for arg in sys.argv[1:]]

	for p in products:
		print(f"{p} = {list_to_str_product(list(factors(p)))}")
