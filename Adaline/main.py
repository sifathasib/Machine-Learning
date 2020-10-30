import pandas as pd
from adaline import AdalineGD
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
df = pd.read_csv('D:\VS Code projects\Machine Learning\data\iris.data',header=None)
#print(df[50:60])

# select setos and versicolor 
y = df.iloc[0:100,4].values  
y = np.where(y=='Iris-setosa',1,-1)
X = df.iloc[0:100,[0,2]].values

def plot_decision_region(X,y,classifier,resolution=0.02):
    markers =('s','x','o','^','v')
    colors = ('red','blue','lightgreen','gray','cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    x_min ,x_max = X[:,0].min()-1,X[:,0].max()+1
    y_min ,y_max = X[:,1].min()-1,X[:,1].max()+1
    xx,yy = np.meshgrid(np.arange(x_min,x_max,resolution),np.arange(y_min,y_max,resolution))
    Z = classifier.predict(np.array([xx.ravel(),yy.ravel()]).T)
    Z = Z.reshape(xx.shape)
    plt.contourf(xx,yy,Z,alpha=0.3,cmap= cmap)
    plt.xlim(xx.min(),xx.max())
    plt.ylim(yy.min(),yy.max())
    
    for idx,cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl,0],y=X[y==cl,1],
                    alpha =0.8,
                    c= colors[idx],
                    marker=markers[idx],
                    label=cl,edgecolor="black")
        



fig ,ax = plt.subplots(nrows=1, ncols=2, figsize=(10,4))
ada1 = AdalineGD(eta = 0.01,n_itr = 10).fit(X,y)
ax[0].plot(range(1,len(ada1.cost_)+1),np.log10(ada1.cost_),marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(sum_squarred error)')
ax[0].set_title('Adaline learning rate 0.01')

ada2 = AdalineGD(eta= 0.00001,n_itr= 10).fit(X,y)
ax[1].plot(range(1,len(ada2.cost_)+1),np.log10(ada2.cost_),marker='o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('log(sum squarred errors)')
ax[1].set_title('Adaline learning rate 0.00001')
plt.show()


X_std = np.copy(X)
X_std[:,0] = (X_std[:,0]-X_std[:,0].mean())/X_std[:,0].std()
X_std[:,1] = (X_std[:,1]-X_std[:,1].mean())/X_std[:,1].std()
ada3 = AdalineGD(eta = 0.1,n_itr= 10).fit(X_std,y)
plot_decision_region(X_std,y,classifier=ada3)
plt.title('adaline gradient descent')
plt.xlabel('sepal lenght [std]')
plt.ylabel('petal lenght [std]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()