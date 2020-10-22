import numpy as np 

a = np.array([[1.0,2.0],[3.0,4.0]])
print(a)
print(a.transpose())
print(np.linalg.inv(a))
u = np.eye(2)
print(u)

j= np.array([[0.0,-1.0],[1.0,0.0]])
print(j)
print(j @ j) #matrix product

print(np.linalg.eig(j))