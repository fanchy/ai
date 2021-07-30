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

class GSKernal:
    def __init__(self):
        self.sigma = 10
    def cal(self, x, z):
        result = None
        if len(x.shape) == 2:
            result = np.linalg.norm(x-z, axis=1) ** 2
        else:
            result = np.linalg.norm(x-z) ** 2
        return np.exp(-1 * result / (2 * self.sigma**2))
class SVM:
    def __init__(self, kn = None):
        self.w = None #w
        self.b = None #b
        if not kn:
            kn = GSKernal()
        self.kn= kn
        return
    def isSatisfyKKT(self, i):
        '''
        查看第i个α是否满足KKT条件
        :param i:α的下标
        :return:
            True：满足
            False：不满足
        '''
        gxi =self.calc_gxi(i)
        yi = self.trainLabelMat[i]

        #判断依据参照“7.4.2 变量的选择方法”中“1.第1个变量的选择”
        #式7.111到7.113
        #--------------------
        #依据7.111
        if (math.fabs(self.alpha[i]) < self.toler) and (yi * gxi >= 1):
            return True
        #依据7.113
        elif (math.fabs(self.alpha[i] - self.C) < self.toler) and (yi * gxi <= 1):
            return True
        #依据7.112
        elif (self.alpha[i] > -self.toler) and (self.alpha[i] < (self.C + self.toler)) \
                and (math.fabs(yi * gxi - 1) < self.toler):
            return True

        return False
    def fit(self, X, Y):
        scalerX = sklearn.preprocessing.StandardScaler().fit(X)#StandardScaler
        #Y = Y.reshape(-1, 1)
        X = scalerX.transform(X)
        
        w = np.zeros(X.shape[1])
        b = 0
        a = np.zeros(X.shape[0])
        i = 0
        j = 19
        y1 = Y[i]
        y2 = Y[j]
        #print(y1, y2)    
        x1 = X[i]
        x2 = X[j]
        k11 = self.kn.cal(x1, x1)
        k12 = self.kn.cal(x1, x2)
        k22 = self.kn.cal(x2, x2)
              
        ay = np.multiply(a, Y)
        ayk1 = np.dot(ay, self.kn.cal(X, x1))
        ayk2 = np.dot(ay, self.kn.cal(X, x2))
        a2_old = a[j]
        a2_new = a2_old + (y2 * (ayk1 - y1 - (ayk2 - y2))) / (k11 - 2*k12 + k22)
        a[j] = a2_new
        print('data', X.shape, Y.shape, w.shape, k11, k12, k22, ayk1, ayk2, a2_new)
        #print(a)
        #self.kn()
        return
        

if __name__ == '__main__':
    svm = SVM()
    svm.fit(dataset.data, dataset.target)
