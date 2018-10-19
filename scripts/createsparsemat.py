# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 18:36:57 2018

@author: Nitin
"""

from scipy.sparse import csr_matrix
import numpy as np
from sklearn.datasets import dump_svmlight_file
import argparse
import os
#num_features = 8315




def svm2mat(filename, num_features):
    libsvmContents = open(filename).readlines()
    #Create a matrix of num_train x num_features
    num_train = len(libsvmContents)
    labels = []
    arr = np.zeros((num_train,num_features)).astype(float)
    for i,line in enumerate(libsvmContents):
        contents = line.split()[1:]
        label = int(line.split()[0])
        locations = [int(item.split(":")[0]) for item in contents]
        values = [float(item.split(":")[1]) for item in contents]
        for loc in locations:
            for val in values:
                arr[i][loc-1] = val
        labels.append(label)
    return np.array(arr), np.array(labels)

def refactor_libsvm_file(trainFile, testFile,num_features):
    x,y = svm2mat(trainFile,  num_features)
    x_test,y_test = svm2mat(testFile,  num_features)

    dump_svmlight_file(x,y,trainFile,zero_based = False)
    dump_svmlight_file(x_test,y_test,testFile,zero_based=False)
    x = csr_matrix(x)
    x_test = csr_matrix(x_test)
    sparsity = np.count_nonzero(x.toarray())
    sparsity += np.count_nonzero(x_test.toarray())
    sparsity /= x.shape[0]*x.shape[1] + x_test.shape[0]*x_test.shape[1]
    print("Dumped")


if __name__ ==  "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_folder", help="Dataset name as found in ./data/",
        type = str,required =True)
    parser.add_argument("--num_features", help="Total number of features",
        type = int,required =True)

    args = parser.parse_args()
    num_features = args.num_features
    ##Reuters
    #data_folder = "/user/nitinnat/Pegasos4/dsvm/peersim-pegasos/data/"
    dataset = args.dataset_folder.split("/")[-1].split("\\")[-1]
    trainFile =  dataset + "-train.dat"
    testFile = dataset + "-test.dat"
    trainFile = os.path.join(args.dataset_folder,trainFile)
    testFile = os.path.join(args.dataset_folder,testFile)
    refactor_libsvm_file(trainFile,testFile,num_features)
    
