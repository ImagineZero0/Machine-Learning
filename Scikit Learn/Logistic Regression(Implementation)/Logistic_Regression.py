'''
Varun Gupta
This is a implemention of the simple Logistic regression using the python OOP approach having the required methods for fit and predict and the required methods including sigmoid activation function.
Machine Learning Repository
'''
import numpy as np

class LogisticRegressionGD(object):
    """
        Logistic Regression classifier using Gradient Descent.
        
        Parameters:
            eta : float
                Learning rate
            n_iter : int
                Passes over the training dataset.
            random_state : int
                Random number generator seed for random weight initialization.
                
        Attributes:
            w_ : 1-d array
                Weights after fitting.
            
            cost_ : list
                Sum of squares cost function value in each epoch.

    """
    def __init__(self,eta=0.05,n_iter=100,random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        
    def fit(self,X,y):
        """ Fit training data.
            
            Parameters:
                X : {array-like}, shape = [n_samples, n_features]
                    Training vectors, where n_samples is the number of samples and 
                    n_features is the number of features.
                y : array-like, shape = [n_samples]
                    Target values.
                    
            
            Returns:
                self : object
        
        """
        
        rand_num = np.random.RandomState(self.random_state)
        self.w_ = rand_num.normal(loc = 0.0,scale = 0.01,size = 1 + X.shape[1])        #Selecting the weight around a normal distribution
        
        self.cost_ = []             # An array to store the cost after every iteration and check that the cost is decreasing
        
        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta*errors.sum()
            
            cost = (-y.dot(np.log(output)) - ((1-y).dot(np.log(1-output))))
            self.cost_.append(cost)
            return self
        

    def net_input(self,X):
        """ Calculate net input"""
        return np.dot(X,self.w_[1:]) + self.w_[0]

    def activation(self,z):
        """ Compute logistic sigmoid activation"""
        #using the clipping function as there is no need for it because the values larger or smaller would be very close to zero
        return 1./(1. + np.exp(-np.clip(z,-250,250))) 

    def predict(self,X):
        """ Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0,1,0)