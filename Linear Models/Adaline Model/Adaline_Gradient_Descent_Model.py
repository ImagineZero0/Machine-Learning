'''
Varun Gupta
An implementation of Adaline Gradient Descent Model using the Python OOP approach using a python class, which initializes a model with the predict and fit method in which the fit method is a improvement over the Rosenblatt's Perceptron as it uses the linear function during the weights update and threshold function for calculating the class labels.
Machine Learning Repository
'''

import numpy as np

class AdadlineGD(object):
    """
        ADAptive LInear NEuron classifier.
        
        Parameters:
            eta : float
                Learning Rate
            n_iter : int
                Passes over the training dataset.
            random_state : int
                Random number generator seed for random weight initialization.
                
        Attributes:
            w_ : 1d-array
                Weights after fitting the model using training dataset
            cost_ : list
                Sum-of-Squares (SSE) cost function value in each epoch.
            
    """
    
    def __init__(self,eta = 0.01,n_iter = 50,random_state = 1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        
    def fit(self,X,y):
        """
            Fitting the training data.
            
            Parameters:
                X : {array-like}, shape = [n_samples, n_features]
                    Training vectors, where n_samples is the number of samples and 
                    n_features is the number of features
                y : {array-like}, shape = [n_samples]
                    Target values.
                
            Returns:
                self : object
                
        """
        
        rand_num = np.random.RandomState(self.random_state)
        self.w_ = rand_num.normal(loc = 0.0, scale = 0.01,size = 1 + X.shape[1])
        self.cost_ = []
        
        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum()/2.0
            self.cost_.append(cost)
        return self
    
    
    def net_input(self, X):
        """ Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """ Compute linear activation """
        # just to illustrate the linear function during the weight updates
        return X
    
    def predict(self,X):
        """ Return class label after unit step"""
        return np.where(self.activation(self.net_input(X)) >= 0.0 , 1, -1)
    