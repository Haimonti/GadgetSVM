# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 02:07:31 2018

@author: Nitin
"""
import argparse
import numpy as np
from tqdm import tqdm
#parser = argparse.ArgumentParser()
#parser.add_argument("--filepath", help="Path to libsvm formatted file.",type = str,
#	 nargs='?', const=1, required=True)

#filepath = parser.parse_args().filepath

def normalize(arr):
    return list(np.array((arr - np.min(arr))/(np.max(arr) - np.min(arr))))
#Takes a file in sparse libsvm format and normalizes it
trainpath = './covtype_bn_train.dat'
testpath = './covtype_bn_test.dat'


f = open(testpath)
contents = f.read().split('\n')
n = len(contents)
print("%d datapoints found in file."%n)
labels = []
nonzero_indices = []
features = []
new_output = []

for i,string in tqdm(enumerate(contents)):
    if string != '':
        splits = string.split()
        
        temp = splits[0]
        temp1 = [item.split(':')[0] + ':' for item in splits[1:]]
        temp2 = normalize([float(item.split(':')[1]) for item in splits[1:]])
        
        labels.append(temp)
        nonzero_indices.append(temp1)
        features.append(temp2)
        new_string = temp + ' ' + ' '.join(temp1[k] + str(temp2[k]) for k in range(len(temp1)))
        new_output.append(new_string)
    else:
        print("Empty line found at index %d. "%i)
f.close()
new_output = '\n'.join(new_output)

#with open(trainpath,'w') as f:
#    f.write(normalize_libsvm_format(trainpath))

with open(testpath,'w') as wf:
    wf.write(new_output)



