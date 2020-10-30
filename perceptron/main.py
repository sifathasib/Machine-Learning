import pandas as pd  
import matplotlib.pyplot as plt  
import numpy as np  
from perceptron import Perceptron
from matplotlib.colors import ListedColormap
df = pd.read_csv('D:\VS Code projects\Machine Learning\data\iris.data',header=None)
#print(df[50:60])

# select setos and versicolor 
y = df.iloc[0:100,4].values  
y = np.where(y=='Iris-setosa',1,-1)
X = df.iloc[0:100,[0,2]].values

ppn = Perceptron(eta = 0.01,n_itr=100)
ppn.fit(X,y)

def plot_decisions_region(X,y,classifier,resolution=0.02):
    markers= ('s','x','o','^','v')
    colors = ('green','blue','gray','cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    x_min,x_max = X[:,0].min()-1,X[:,0].max()+1
    y_min,y_max = X[:,1].min()-1,X[:,1].max()+1
    xx,yy = np.meshgrid(np.arange(x_min,x_max,resolution),np.arange(y_min,y_max,resolution))
    Z = classifier.predict(np.array([xx.ravel(),yy.ravel()]).T)
    Z =Z.reshape(xx.shape)
    plt.contourf(xx,yy,Z,alpha=0.3,cmap=cmap)
    plt.xlim(xx.min(),xx.max())
    plt.ylim(yy.min(),yy.max())
    #plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl,
                    edgecolor='black')
    '''plt.scatter(X[:50,0],X[:50,1],color='red',marker='x',label='setosa')
    plt.scatter(X[50:100,0],X[50:100,1],color='blue',marker='o',label='versicolor')'''
    
    
plot_decisions_region(X,y,classifier=ppn)    
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.show()
'''plt.figure(1)
plt.plot(range(1,len(ppn.errors_)+1),ppn.errors_,marker='o')
plt.xlabel('epochs')
plt.ylabel('number of updates')
plt.show()
plt.close()
plt.figure(2)
plt.scatter(X[:50,0],X[:50,1],color='red',marker='x',label='setosa')
plt.scatter(X[50:100,0],X[50:100,1],color='blue',marker='o',label='versicolor')
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.show()
plt.close()'''
