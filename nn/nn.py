
import numpy as np 
import sklearn
from sklearn import datasets
from sklearn.datasets import load_breast_cancer
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


dataset = None
DT_FLAG = 0 #1测试预测，2测试回归拟合

if DT_FLAG:
    dataset = sklearn.datasets.load_breast_cancer()
else:
    dataset = sklearn.datasets.load_boston()

class ReLU:
    def __init__(self):
        return
    def cal(self, z):
        return np.clip(z, 0, np.inf)
    def grad(self, x):
        return (x > 0).astype(int)
class Sigmoid:
    def __init__(self):
        return
    def cal(self, z):
        return 1 / (1 + np.exp(-z))
    def grad(self, x):
        z = self.cal(x)
        return z*(1-z)

class LossMSE:
    def __init__(self):
        return
    def loss(self, y_pred, y):
        return np.sum(np.multiply((y_pred - y) , (y_pred - y))) / 2 /int(y.shape[0]) 
    def grad(self, y_pred, y):
        return (y_pred - y)/y.shape[0]
class LossCrossEntropy:
    def loss(self, y_pred, y):
        eps = np.finfo(float).eps
        cross_entropy = -np.sum(y * np.log(y_pred + eps))
        return cross_entropy
    def grad(self, y_pred, y):
        grad = y_pred - y
        return grad

class LinearOut:#如果用于回归，只是线性输出，不做任何转换，为了可以有统一的结构
    def cal(self, z):
        return z
    def grad(self, x):
        self.gradCache = np.ones(np.shape(x))
        return self.gradCache

class NN:
    def __init__(self):
        self.Wh = None #隐藏层参数w
        self.Wo = None #输出层参数w
        self.numHideUnit = 5
        self.funcActivation = ReLU() #隐藏层激活函数
        if DT_FLAG:
            self.lossFunc = LossMSE()
            self.funcOut = Sigmoid()
        else:
            self.lossFunc = LossMSE()
            self.funcOut = LinearOut()
        return
    def fit(self, X, Y):
        scalerX = sklearn.preprocessing.StandardScaler().fit(X)#StandardScaler
        Y = Y.reshape(-1, 1)
        X = scalerX.transform(X)
        if not DT_FLAG:
            scalerY = sklearn.preprocessing.StandardScaler().fit(Y)#MinMaxScaler
            Y = scalerY.transform(Y)

        m, n = np.shape(X)

        funcActivation = self.funcActivation
        funcOut = self.funcOut 
        # 10 1000 0.01 for breat cancer
        numLayerHide = 5
        maxIterTimes = 2000 
        eta  = 0.01
        if DT_FLAG:
            eta  = 10
        else:
            #maxIterTimes = 100000
            eta  = 0.5#eta  = 0.00001
            pass

        outputNodeNum = 1
        Wh = np.random.rand(n, numLayerHide) * 0.01 
        Bh = np.random.rand(numLayerHide) * 0.01 
        Wo = np.random.rand(numLayerHide, outputNodeNum) * 0.01 #Why +1? for reserver b 
        Bo = np.random.rand(outputNodeNum) * 0.01 

        errLog = []
        dO_Old = 0
        dH_Old = 0
        for i in range(maxIterTimes):
            #forward
            Lh = np.dot(X, Wh) + Bh            
            Yh = funcActivation.cal(Lh) #隐藏层输出

            Lo = np.dot(Yh, Wo) + Bo
            Yo = funcOut.cal(Lo)

            loss = self.lossFunc.loss(Yo, Y)
            errLog.append(loss)
            if loss < 0.001:
                print('fit finish', i)
                break
            
            delta = np.multiply(self.lossFunc.grad(Yo, Y) , funcOut.grad(Lo))

            dBo  = delta
            dWo  = np.dot(Yh.T, dBo)
            
            dBh = np.multiply(np.dot(delta, Wo.T) , funcActivation.grad(Lh))
            dWh = np.dot(X.T, dBh)

            Wo = Wo - eta * dWo
            Wh = Wh - eta * dWh
            Bo = Bo - eta * dBo
            Bh = Bh - eta * dBh
            
        score = 0
        if DT_FLAG == 0:
            score = r2_score(Yo, Y)
        else:
            score = r2_score((Yo > 0.5).astype(int), Y)
        print('score', score)
        return
    def fitNoB(self, X, Y):
        m, n = np.shape(X)
        scalerX = preprocessing.StandardScaler().fit(X)#StandardScaler
        Y = Y.reshape(-1, 1)
        X = scalerX.transform(X)
        if not DT_FLAG:
            scalerY = preprocessing.StandardScaler().fit(Y)#MinMaxScaler
            Y = scalerY.transform(Y)

        allOneCol = np.ones(m)
        X = np.insert(X, n, values=allOneCol, axis=1)#add all 1 col
        
        m, n = np.shape(X)
        
        funcActivation = self.funcActivation
        funcOut = self.funcOut 
        # 10 1000 0.01 for breat cancer
        numLayerHide = 10
        maxIterTimes = 3000 
        eta  = 0.01
        if DT_FLAG:
            eta  = 2#0.01
        else:
            #maxIterTimes = 100000
            eta  = 0.001#eta  = 0.00001
            pass
        
        Wh = np.random.rand(n, numLayerHide) * 0.01 
        Wo = np.random.rand(numLayerHide + 1, 1) * 0.01 #Why +1? for reserver b 

        errLog = []
        dO_Old = 0
        dH_Old = 0
        for i in range(maxIterTimes):
            #forward
            Lh = np.dot(X, Wh)
            Yh = funcActivation.cal(Lh) #隐藏层输出
            Yh = np.insert(Yh, np.shape(Yh)[1], values=allOneCol, axis=1)#add all 1 col
            Lo = np.dot(Yh, Wo)
            Yo = funcOut.cal(Lo)
            loss = self.lossFunc.loss(Yo, Y)
            errLog.append(loss/X.shape[0])
            if loss < 0.001:
                print('fit finish', i)
                break
            #backword
            delta = np.multiply(self.lossFunc.grad(Yo, Y) , funcOut.grad(Lo))
            dO  = np.dot(Yh.T, delta)
            dH = np.dot(X.T, np.multiply(np.dot(delta, Wo.T[:,:-1]) , funcActivation.grad(Lh)))
            Wo = Wo - eta * dO
            Wh = Wh - eta * dH

        score = 0
        if DT_FLAG == 0:
            score = r2_score(Yo, Y)
        else:
            score = r2_score((Yo > 0.5).astype(int), Y)
            
        print(errLog[-100:])
        print(Yo[0:20,])
        print('*'*20)
        print( Y[0:20,])
        print('score', score)
        return

if __name__ == '__main__':
    nn = NN()
    print('data', dataset.data.shape)
    nn.fit(dataset.data, dataset.target)
