# -*- coding: utf8 -*-
import numpy as np 
import sklearn
from sklearn import datasets
from sklearn.datasets import load_breast_cancer
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
dataset = sklearn.datasets.load_breast_cancer()
dataset.target[dataset.target == 0] = -1 #0/1 标记替换成1/-1标记

class SVM:
    def __init__(self):
        self.w = None #w
        self.b = None #b
        return
    def fit(self, X, Y):
        scalerX = sklearn.preprocessing.StandardScaler().fit(X)#StandardScaler
        Y = Y.reshape(-1, 1)
        X = scalerX.transform(X)
        
        w = np.zeros((1, X.shape[1]))
        b = 0
        print('data', X.shape, Y.shape, w.shape)
        return
        

if __name__ == '__main__':
    svm = SVM()
    svm.fit(dataset.data, dataset.target)
