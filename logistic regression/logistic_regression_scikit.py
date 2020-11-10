from sklearn import datasets  
import numpy as np  
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import StandardScaler   
from sklearn.linear_model import LogisticRegression  
import matplotlib.pyplot as plt   
from plotting import plot_decision_region      
#dataset loading    
iris = datasets.load_iris()
X = iris.data[:,[2,3]]
y = iris.target   
#splitting the dataset           
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state=1,stratify=y)
# standardization  
sc= StandardScaler()
sc.fit(X_train)  
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

#algorithm  
lr= LogisticRegression(C=100,random_state=1)
lr.fit(X_train_std,y_train)
'''lr.predict_proba(X_test_std[:3,:])
lr.predict_proba(X_test_std[:3,:]).sum(axis=1)
lr.predict(X_test_std[:3,:])
print(lr.coef_[0])'''
# plotting 
X_combined_std = np.vstack((X_train_std,X_test_std))
y_combined = np.hstack((y_train,y_test))

plot_decision_region(X=X_combined_std,y=y_combined,classifier = lr ,test_idx=range(105,150))
plt.xlabel('petal length standardized')
plt.ylabel('petal width standardized')
plt.legend(loc='upper left')
plt.show()

