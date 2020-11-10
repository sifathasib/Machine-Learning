import numpy as np 

np.random.seed(1)
x = np.array([[1,2],[-4,5],[-5,1]])
y = np.logical_xor(x[:,0]>0,x[:,1]>0)
y = np.where(y,1,-1)
print(y)
