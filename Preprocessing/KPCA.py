'''
Varun Gupta
This is the implementation of a kernel method used in the Kernel Principal Component Analysis procedure and here we typically implement the Radial Basis Function or Gaussian Kernel to evaluate new features for the provided dataset.
Machine Learning repository
'''

from scipy.spatial.distance import pdist,squareform
from scipy import exp
from scipy.linalg import eigh
import numpy as np

def rbf_kernel_pca(X,gamma,n_components):
    """
        RBF Kernel PCA Implementaion.
        
        Parameters:
            X : {Numpy ndarray}, shape = [n_samples,n_features]
            
            gamma : float
                Tuning parameter of the RBF Kernel
                
            n_components : int
                Number of principal components to return
                
        Returns:
        X_pc: {Numpy ndarray}, shape = [n_samples,k_features]
            Projected dataset
        
        lambdas : list
            Eigenvalues
            
    """
    sq_dists = pdist(X,'sqeuclidean')
    
    mat_sq_dists = squareform(sq_dists)
    
    K = exp(-gamma*mat_sq_dists)
    
    N = K.shape[0]
    one_n = np.ones((N,N))/N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)
    
    eigvals, eigvecs = eigh(K)
    eigvals, eigvecs = eigvals[::-1], eigvecs[:,::-1]
    
    alphas = np.column_stack((eigvecs[:,i] for i in range(n_components)))
    lambdas = [eigvals[i] for i in range(n_components)]
    
    return alphas,lambdas