import pandas as pd   
import numpy as np   
from LogisticRegressionGD import LogisticRegressionGd 
import matplotlib.pyplot as plt  
from matplotlib.colors import ListedColormap  
from sklearn import datasets   
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler

iris = datasets.load_iris()
X = iris.data[:,[2,3]]
y = iris.target
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=1,stratify=y)
sc = StandardScaler()
sc.fit(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)

'''df = pd.read_csv('D:\VS Code projects\Machine Learning\data\iris.data',header=None)
y = df.iloc[0:100,4].values
y = np.where(y=='Iris-setosa',1,-1)
X = df.iloc[0:100,[0,2]].values'''

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
X_train_01_subset = X_train[(y_train==0)|(y_train==1)]
y_train_01_subset = y_train[(y_train==0)|(y_train==1)]
lrgd = LogisticRegressionGd(eta=0.05,n_itr=1000,random_state=1)
lrgd.fit(X_train_01_subset,y_train_01_subset)
plot_decision_region(X=X_train_01_subset,y=y_train_01_subset,classifier=lrgd)
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.legend(loc='upper left')
plt.show()
