import pandas as pd  
import matplotlib.pyplot as plt  
import numpy as np  
from perceptron import Perceptron
df = pd.read_csv('D:\VS Code projects\Machine Learning\data\iris.data',header=None)
print(df[50:60])

# select setos and versicolor 
y = df.iloc[0:100,4].values  
y = np.where(y=='Iris-setosa',1,-1)
X = df.iloc[0:100,[0,2]].values
ppn = Perceptron(eta = 0.1,n_itr=100)
ppn.fit(X,y)
plt.figure(1)
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
plt.close()
