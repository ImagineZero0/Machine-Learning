'''
Varun Gupta
Here we just implement our normal ANN Model that contains the respective functions to fully evaluate and train weights for function approximation with the gradient descent optimization approach to reach the global optimal solution. We have respectively implemented the backpropragation algorithm for the ANN so that it henceforth learns the weights in the best optimal way.
Machine Learning Repository
'''

import numpy as np
import sys

class NeuralNetMLP(object):
    """
        Feedforward neural network / Multi-layer perceptron
        classifier.
        
        Parameters:
            n_hidden : int (default : 30)
                Number of hidden units
            l2 : float (default: 0.)
                Lambda value for L2-regularrization
                No regularization if l2=0. (default)
            epochs : int (default: 100)
                Number of passes over the training set.
            eta : float (default : 0.001)
                Learning rate.
            shuffle : bool (default : True)
                Shuffles training data every epoch
                if True to prevent circles.
            minibatch_size : int (default: 1)
                Number of training samples per minibatch
            seed : int (default : None)
                Random seed for initializing weights and shuffling.
        Attributes:
            eval_ : dict
                Dictionary collecting the cost , training accuracy,
                and validation accuracy for each epoch during training.
    """
    
    def __init__(self, n_hidden = 30,l2 = 0.,epochs = 100,eta = 0.001,shuffle = True,minibatch_size = 1,seed = None):
        self.random = np.random.RandomState(seed)
        self.n_hidden = n_hidden
        self.l2 = l2
        self.epochs = epochs
        self.eta = eta
        self.shuffle = shuffle
        self.minibatch_size = minibatch_size
        
    def _onehot(self,y,n_classes):
        """
            Encode labels into one-hot representation.
            
            Parameters:
                y : array, shape = [n_samples]
                Target values.
            Returns:
                oneshot : array, shape = [n_samples,n_labels]
        """
        onehot = np.zeros((n_classes,y.shape[0]))
        for idx , val in enumerate(y.astype(int)):
            onehot[val,idx] = 1.
        return onehot.T
    
    def _sigmoid(self,z):
        """
            Compute logistic function (sigmoid)
        
        """
        return 1./(1. + np.exp(-np.clip(z,-250,250)))
    
    def _forward(self,X):
        """
            Compute forward propagation step.
        """
        z_h = np.dot(X,self.w_h) + self.b_h
        
        a_h = self._sigmoid(z_h)
        
        z_out = np.dot(a_h,self.w_out) + self.b_out
        
        a_out = self._sigmoid(z_out)
        
        return z_h,a_h,z_out,a_out
     
    def _compute_cost(self,y_enc,output):
        """
            Compute cost function.
            
            Parameters :
                y_enc : array, shape = (n_samples,n_labels)
                    one-hot encoded class labels.
                output : array, shape = [n_samples, n_output_units]
            Returns:
                cost : float
                    Regularized cost
        """
        L2_term = (self.l2*(np.sum(self.w_h**2.) +np.sum(self.w_out**2.)))
        term1 = -y_enc*(np.log(output))
        term2 = (1. - y_enc)*np.log(1. - output)
        cost = np.sum(term1 - term2) + L2_term
        return cost
    
    def predict(self,X):
        """
            Predict class labels.
            
            Parameters:
                X : array,shape = [n_samples,n_features]
                    Input layer with original features.
            Returns :
                y_pred : array, shape = [n_samples]
                    Predicted class labels.
        """
        z_h, a_h,z_out,a_out = self._forward(X)
        y_pred = np.argmax(z_out,axis = 1)
        return y_pred
    
    def fit(self,X_train,y_train,X_valid,y_valid):
        """
            Learn weights from training data.
            
            Parameters:
                X_train : array, shape = [n_samples,n_features]
                    Input layer with original features.
                y_train : array, shape = [n_samples]
                    Target class labels.
                X_valid : array, shape = [n_samples,n_features]
                    Sample features for validation during training.
                y_valid : array, shape = [n_samples]
                    Sample labels for validation during training.
            Returns:
                self
        """
        n_output = np.unique(y_train).shape[0]
        
        n_features = X_train.shape[1]
        
        self.b_h = np.zeros(self.n_hidden)
        
        self.w_h = self.random.normal(loc = 0.0,scale= 0.1,size = (n_features,self.n_hidden))
        
        self.b_out = np.zeros(n_output)
        
        self.w_out = self.random.normal(loc = 0.0,scale = 0.1,size = (self.n_hidden,n_output))
        
        epoch_strlen = len(str(self.epochs))
        
        self.eval_ = {'cost':[],'train_acc':[],'valid_acc':[]}
        
        y_train_enc = self._onehot(y_train,n_output)
        
        for i in range(self.epochs):
            indices = np.arange(X_train.shape[0])
            
            if self.shuffle:
                self.random.shuffle(indices)
                
            for start_idx in range(0,indices.shape[0]-self.minibatch_size + 1,self.minibatch_size):
                batch_idx = indices[start_idx:start_idx+self.minibatch_size]
                
                z_h , a_h , z_out , a_out = self._forward(X_train[batch_idx])
                
                
                sigma_out = a_out - y_train_enc[batch_idx]
                
                sigmoid_derivative_h = a_h*(1. - a_h)
                
                sigma_h = (np.dot(sigma_out,self.w_out.T)*sigmoid_derivative_h)
                
                grad_w_h = np.dot(X_train[batch_idx].T,sigma_h)
                grad_b_h = np.sum(sigma_h,axis = 0)
                
                grad_w_out = np.dot(a_h.T,sigma_out)
                grad_b_out = np.sum(sigma_out,axis = 0)
                
                delta_w_h = (grad_w_h+self.l2*self.w_h)
                delta_b_h = (grad_b_h)
                self.w_h -= self.eta*delta_w_h
                self.b_h -= self.eta*delta_b_h
                
                delta_w_out = (grad_w_out+self.l2*self.w_out)
                delta_b_out = grad_b_out
                self.w_out -= self.eta*delta_w_out
                self.b_out -= self.eta*delta_b_out
                
            z_h , a_h , z_out , a_out = self._forward(X_train)
            
            cost = self._compute_cost(y_enc = y_train_enc,output = a_out)
            
            y_train_pred = self.predict(X_train)
            y_valid_pred = self.predict(X_valid)
            
            train_acc = ((np.sum(y_train == y_train_pred)).astype(np.float)/X_train.shape[0])
            valid_acc = ((np.sum(y_valid == y_valid_pred)).astype(np.float)/X_valid.shape[0])
            
            sys.stderr.write('\r%0*d/%d | Cost: %.2f | Train/Valid Acc. : %.2f%%/%.2f%%' %(epoch_strlen,i+1,self.epochs,cost,train_acc*100,valid_acc*100))
            
            sys.stderr.flush()
            
            self.eval_['cost'].append(cost)
            self.eval_['train_acc'].append(train_acc)
            self.eval_['valid_acc'].append(valid_acc)
            
        return self