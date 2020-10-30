''' this module has nothing to do with 
main file. it's built for just understanding some things about
to implement the code '''  

import numpy as np  
x = np.array([[1,2],[2,3],[3,4]])
w = np.array([2,2])
y = np.array([1,2,3])
output= np.dot(x,w)
errors = y - output
et = np.array([1,1,1])
print(x)
print(x.T)
print(et)
print(np.dot(x.T,et))