import pandas as pd  
import numpy as np  
from AdalineSGD import AdalineSGD  
import matplotlib.pyplot as plt  
from matplotlib.colors import ListedColormap  

df = pd.read_csv('D:\VS Code projects\Machine Learning\data\iris.data',header =None)
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

X_std = np.copy(X)
X_std[:,0] = (X_std[:,0]-X_std[:,0].mean())/X_std[:,0].std()
X_std[:,1] = (X_std[:,1]-X_std[:,1].mean())/X_std[:,1].std()

ada = AdalineSGD(eta=0.01,n_itr= 15,random_state=1).fit(X_std,y)
#ada = AdalineSGD(eta=0.01,n_itr= 15,random_state=1).partial_fit(X_std[0:30,:],y[0:30])
plot_decision_region(X_std,y,classifier= ada)
plt.title('adaline stochastic gradient descent')
plt.xlabel('sepal length std')
plt.ylabel('petal length std')
plt.legend(loc = 'upper left')
plt.show()
plt.plot(np.arange(1,len(ada.cost_)+1),ada.cost_,marker='o')
plt.xlabel('epochs')
plt.ylabel('Avg cost')
plt.show()
