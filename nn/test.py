

import numpy as np 
from sklearn.datasets import  fetch_california_housing
from sklearn.datasets import load_breast_cancer
dataset = load_breast_cancer()
X = dataset.data
y = dataset.target
print(type(X))
print(type(y))
print(y.shape)