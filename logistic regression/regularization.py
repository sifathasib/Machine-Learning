from sklearn import datasets   
import numpy as np  
import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LogisticRegression   
iris = datasets.load_iris()
X =iris.data[:,[1,2]]
y = iris.target   

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size= 0.3,random_state=1,stratify=y)
sc = StandardScaler()
sc.fit(X_train) 
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

weights,params = [],[]
for c in np.arange(-5,5):
    lr = LogisticRegression(C = 10.**c,random_state=1)
    lr.fit(X_train_std,y_train) 
    weights.append(lr.coef_[1])
    params.append(10.**c)
weights = np.array(weights)
plt.plot(params,weights[:,0],label='petal length',)
plt.plot(params,weights[:,1],label='petal width',linestyle='--',)
plt.xlabel('C')
plt.ylabel('weigths')
plt.legend(loc='upper left')
plt.xscale('log')
plt.show()



