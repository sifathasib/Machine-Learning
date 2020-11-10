import numpy as np   
import matplotlib.pyplot as plt  
from matplotlib.colors import ListedColormap   
def plot_decision_region(X,y,classifier,test_idx=None,resolution=0.02):
    markers = ('s','x','o','^','v')
    colors = ('red','blue','lightgreen','gray','cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    x_min,x_max = X[:,0].min()-1,X[:,0].max()+1
    y_min,y_max = X[:,1].min()-1,X[:,1].max()+1
    xx,yy = np.meshgrid(np.arange(x_min,x_max,resolution),np.arange(y_min,y_max,resolution))
    Z = classifier.predict(np.array([xx.ravel(),yy.ravel()]).T)
    Z = Z.reshape(xx.shape)
    plt.contourf(xx,yy,Z,alpha=0.3,cmap= cmap)
    plt.xlim(xx.min(),xx.max())
    plt.ylim(yy.min(),yy.max())
    
    for idx,cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl,0],y=X[y==cl,1],
                    alpha = 0.8,
                    c= colors[idx],
                    marker = markers[idx],
                    label=cl,edgecolor='black')

    if test_idx:
        X_test,y_test = X[test_idx,:],y[test_idx]
        plt.scatter(X_test[:,0],X_test[:,1],
                    c='',edgecolor='black',alpha=1.0,linewidth=1,
                    marker='o',s=100,label='test set')