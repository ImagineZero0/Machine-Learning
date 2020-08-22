'''
Varun Gupta
This is the implementation of how to plot a decision boundary for a given space of examples and variables and hence is plots the decision boundary between different classification based on the classifier provided.
This function can be used only for the cases specified in the provided jupyter notebooks for different cases concerning more dimensions or parameters would have to change the funciton.
Machine Learning Repository
'''

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np

def plot_decision_boundary(X,y,classifier,test_idx = None,resolution = 0.02):
    markers = ('s','o','x','^','v')
    colors = ('red','blue','lightgreen','gray','cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    x1_min , x1_max = X[:,0].min() - 1 , X[:,0].max() + 1
    x2_min , x2_max = X[:,1].min() - 1 , X[:,1].max() + 1
    xx1 ,xx2 = np.meshgrid(np.arange(x1_min,x2_max,resolution),
                           np.arange(x2_min,x2_max,resolution))
    
    Z = classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1,xx2,Z,alpha = 0.3,cmap = cmap)
    plt.xlim([xx1.min(),xx1.max()])
    plt.ylim([xx1.min(),xx2.max()])
    
    for idx,cl in enumerate(np.unique(y)):
        plt.scatter(x = X[y==cl,0],y= X[y==cl,1],label = cl,c = colors[idx],marker = 
                    markers[idx],edgecolors = 'black',alpha = 0.8)
        
    if test_idx:
        X_test , y_test = X[test_idx,:] , y[test_idx]
        plt.scatter(X_test[:,0],X_test[:,1],c='None',edgecolors = 'black',
                    label = 'test set',s=100,linewidths = 1)