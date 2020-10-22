import numpy as np 
rg = np.random.default_rng(10)

a = np.floor(10*rg.random((3,4)))
print(a)
#a = a.ravel()
#print(a)

a = a.reshape((6,2))
print(a)

a = a.T
print(a)
print(a.shape)