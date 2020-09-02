'''
Varun Gupta
This is a simple implementation of a Simple Linear Regression Model which uses Gradient Descent as an Optimization and SSE as a Cost for the function to approximate the datapoints in the model and hence use the appromixated function to the predict the respective new datapoints and we have defiend the respective methods for this class.
Machine Learning Repository
'''

import numpy as np

class LinearRegressionGD(object):
    def __init__(self,eta = 0.001,n_iter = 20):
        self.eta = eta
        self.n_iter = n_iter
        
    def fit(self,X,y):
        self.w_ = np.zeros(1+X.shape[1])
        self.cost_ = []
        
        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            self.w_[1:] += self.eta*X.T.dot(errors)
            self.w_[0] += self.eta*errors.sum()
            cost = (errors**2).sum()/2.0
            self.cost_.append(cost)
            
        return self
    
    def net_input(self,X):
        return np.dot(X,self.w_[1:]) + self.w_[0]
    
    def predict(self,X):
        return self.net_input(X)