import pandas as pd
from adaline import AdalineGD
import matplotlib.pyplot as plt
import numpy as np
df = pd.read_csv('D:\VS Code projects\Machine Learning\data\iris.data',header=None)
#print(df[50:60])

# select setos and versicolor 
y = df.iloc[0:100,4].values  
y = np.where(y=='Iris-setosa',1,-1)
X = df.iloc[0:100,[0,2]].values

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