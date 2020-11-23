from sklearn import datasets    
import matplotlib.pyplot as plt   
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import StandardScaler  
import numpy as np   
from sklearn.svm import SVC  
from plotting import plot_decision_region 

iris = datasets.load_iris()
X = iris.data[:,[2,3]]
y = iris.target   

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size= 0.3,random_state= 1,stratify = y)
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

#svm = SVC(kernel='linear',C=1.0,random_state=1)
svm = SVC(kernel='rbf',C=1.0,gamma=100,random_state=1)
svm.fit(X_train_std,y_train)

X_combined_std = np.vstack((X_train_std,X_test_std))
y_combined = np.hstack((y_train,y_test))

plot_decision_region(X_combined_std,y_combined,classifier = svm ,test_idx = range(105,150))
plt.xlabel('petal length standardized')
plt.ylabel('petal width standardized')
plt.legend(loc='upper left')
plt.show()