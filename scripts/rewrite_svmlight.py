from sklearn.datasets import load_svmlight_file, dump_svmlight_file
import argparse
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--file", help="path to file",
    type = str,required =True)

args = parser.parse_args()
f = args.file


data = load_svmlight_file(f)
dump_svmlight_file(data[0],data[1], f, zero_based=False)

