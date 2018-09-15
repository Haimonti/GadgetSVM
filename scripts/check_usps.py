# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 00:18:22 2018

@author: Nitin
"""

"""
from sklearn.datasets import load_svmlight_file, dump_svmlight_file
file = "../data/usps/usps-train.dat"
X,y = load_svmlight_file(file)
dump_svmlight_file(X, y, file, zero_based=True, comment=None, query_id=None, multilabel=False)

file = "../data/usps/usps-test.dat"
X,y = load_svmlight_file(file)
dump_svmlight_file(X, y, file, zero_based=True, comment=None, query_id=None, multilabel=False)

for i in range(10):
    file = "../data/usps/t_{}.dat".format(i)
    X,y = load_svmlight_file(file)
    dump_svmlight_file(X, y, file, zero_based=True, comment=None, query_id=None, multilabel=False)
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 23:20:58 2018

File to load and process the USPS data into a binary classification format.
@author: Nitin
"""


import numpy as np
from sklearn.datasets import dump_svmlight_file,load_svmlight_file
from collections import Counter
from sklearn.model_selection import train_test_split
images_tr,labels_tr = load_svmlight_file("usps")
images_tr = images_tr.todense()
labels_tr = np.reshape(np.array([1 if l==8 else -1 for l in labels_tr]),(-1,1))

data_tr = np.hstack((images_tr,labels_tr))
unique, counts = np.unique(labels_tr, return_counts=True)
print("Train Label distribution: " +str(dict(zip(unique, counts))))


images_te, labels_te = load_svmlight_file("usps.t")
images_te = images_te.todense()

labels_te = np.reshape(np.array([1 if l==8 else -1 for l in labels_te]),(-1,1))
#Count number of -1s and number of 1s
unique, counts = np.unique(labels_te, return_counts=True)
print("Test Label distribution: " + str(dict(zip(unique, counts))))

data_te = np.hstack((images_te,labels_te))

#Stack train and test and resplit according to paper
data = np.vstack((data_tr,data_te))
X_train, X_test, y_train, y_test = train_test_split(data[:,0:-1],data[:,-1],random_state = 42, test_size = 0.21176)

def dump_svm_file(data,file):
    data = np.array(data)
    X = data[:, 0:-1]
    y = data[:, -1]
    dump_svmlight_file(X, y, file)

dump_svm_file(np.hstack((X_train,y_train)),'usps.trn' )

dump_svm_file(np.hstack((X_test,y_test)),'usps.tst')