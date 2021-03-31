#!/usr/bin/python3

import json
import sys
import numpy as np

if __name__=='__main__':

	json_fname = sys.argv[1]
	group = sys.argv[2]
	key = sys.argv[3]
	val = sys.argv[4]
	print(json_fname)
	with open(json_fname, "r") as f:
		data = json.load(f)
		data[group][key] = np.int(val)

	with open(json_fname, 'w') as f:
		json.dump(data,f, indent=2)

