# -*- coding: utf8 -*-
import math
import time
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
    def __init__(self, kn = None, C = 1, toler = 0.001):
        self.w = None #w
        self.b = None #b
        if not kn:
            kn = GSKernal()
        self.kn= kn
        self.C = C
        self.toler = toler
        self.alpha = None
        self.X = None
        self.Y = None
        self.kCache = {}
        return
    def calK(self, i, j):
        key = ''
        if i <= j:
            key = '%d_%d'%(i, j)
        else:
            key = '%d_%d'%(j, i)
        ret = self.kCache.get(key)
        if ret:
            return ret
        x1 = self.X[i]
        x2 = self.X[j]
        ret = self.kn.cal(x1, x2)
        self.kCache[key] = ret
        return ret
    def calK2(self, j):
        key = 'All_%d'%(j)
        ret = self.kCache.get(key)
        if ret:
            return ret['data']
        ret = np.zeros(self.X.shape[0])
        x2 = self.X[j]
        for i in range(self.X.shape[0]):
            x1 = self.X[i]
            ret[i] = self.kn.cal(x1, x2)
        self.kCache[key] = {'data':ret}
        return ret
        
    def isZero(self, v):
        return math.fabs(v) < self.toler
    def calGxi(self, i):
        xi = self.X[i]
        ay = np.multiply(self.alpha, self.Y)
        gxi = np.dot(ay, self.calK2(i)) + self.b
        return gxi
    def calcEi(self, i):
        gxi = self.calGxi(i)
        return gxi - self.Y[i]
    def isSatisfyKKT(self, i):
        gxi = self.calGxi(i)
        yi = self.Y[i]
        if self.isZero(self.alpha[i]) and (yi * gxi >= 1):
            return True
        elif self.isZero(self.alpha[i] - self.C) and (yi * gxi <= 1):
            return True
        elif (self.alpha[i] > -self.toler) and (self.alpha[i] < (self.C + self.toler)) \
                and self.isZero(yi * gxi - 1):
            return True

        return False
    def getAlphaJ(self, E1, i):
        E2 = 0
        maxE1_E2 = -1
        maxIndex = -1

        for j in range(self.X.shape[0]):
            if j == i:
                continue
            E2_tmp = self.calcEi(j)
            if E2_tmp == 0:
                continue
            if math.fabs(E1 - E2_tmp) > maxE1_E2:
                maxE1_E2 = math.fabs(E1 - E2_tmp)
                E2 = E2_tmp
                maxIndex = j
        if maxIndex == -1:
            maxIndex = i
            while maxIndex == i:
                maxIndex = int(random.uniform(0, self.X.shape[0]))
            E2 = self.calcEi(maxIndex)

        return E2, maxIndex

    def fit(self, X, Y, iterMax = 10):
        self.kCache.clear()
        scalerX = sklearn.preprocessing.StandardScaler().fit(X)#StandardScaler
        #Y = Y.reshape(-1, 1)
        X = scalerX.transform(X)
        
        w = np.zeros(X.shape[1])
        self.b = 0
        a = np.zeros(X.shape[0])
        self.alpha = a
        self.X = X
        self.Y = Y

        iterStep = 0; parameterChanged = 1

        calTims = 0
        while (iterStep < iterMax) and (parameterChanged > 0):
            iterStep += 1
            parameterChanged = 0

            for i in range(self.X.shape[0]):
                if self.isSatisfyKKT(i):
                    continue
                E1 = self.calcEi(i)
                E2, j = self.getAlphaJ(E1, i)
                                
                y1 = self.Y[i]
                y2 = self.Y[j]
                x1 = self.X[i]
                x2 = self.X[j]
                a1_old = a[i]
                a2_old = a[j]
                
                if y1 != y2:
                    L = max(0, a2_old - a1_old)
                    H = min(self.C, self.C + a2_old - a1_old)
                else:
                    L = max(0, a2_old + a1_old - self.C)
                    H = min(self.C, a2_old + a1_old)
                if L == H:
                    continue
                
                k11 = self.calK(i, i)
                k12 = self.calK(i, j)
                k21 = k12
                k22 = self.calK(j, j)
                      
                ay = np.multiply(a, self.Y)
                ayk1 = np.dot(ay, self.calK2(i))
                ayk2 = np.dot(ay, self.calK2(j))
                a2_new = a2_old + (y2 * (ayk1 - y1 - (ayk2 - y2))) / (k11 - 2*k12 + k22)
                a1_new = a1_old + y1 * y2 * (a2_old - a2_new)
                
                b1New = -1 * E1 - y1 * k11 * (a1_new - a1_old) \
                            - y2 * k21 * (a2_new - a2_old) + self.b
                b2New = -1 * E2 - y1 * k12 * (a1_new - a1_old) \
                        - y2 * k22 * (a2_new - a2_old) + self.b
                bNew = 0
                if (a1_new > 0) and (a1_new < self.C):
                    bNew = b1New
                elif (a2_new > 0) and (a2_new < self.C):
                    bNew = b2New
                else:
                    bNew = (b1New + b2New) / 2

                self.alpha[i] = a1_new
                self.alpha[j] = a2_new
                self.b = bNew
                if math.fabs(a2_new - a2_old) >= 0.00001:
                    parameterChanged += 1
                calTims += 1
                
                if calTims % 50 == 0:
                    print('train iter:%d SMO times:%d'%(iterStep, calTims))
                #time.sleep(1)
                #return
        Yo = self.predict(self.X)
        #print(Yo)
        score = r2_score(Yo, self.Y)
        print('score', score)
        return
    def predict(self, X):
        if len(X.shape) == 1:
            return self.predictOne(X)
        ret = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            ret[i] = self.predictOne(X[i])
        return ret
    def predictOne(self, x):
        ret = np.zeros(self.X.shape[0])
        x2 = x
        for i in range(self.X.shape[0]):
            x1 = self.X[i]
            ret[i] = self.kn.cal(x1, x2)
        ay = np.multiply(self.alpha, self.Y)
        gxi = np.dot(ay, ret) + self.b
        if gxi > 0:
            return 1
        return -1

if __name__ == '__main__':
    svm = SVM()
    print('data', dataset.data.shape)
    svm.fit(dataset.data, dataset.target)
