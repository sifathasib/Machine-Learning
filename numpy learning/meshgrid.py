import numpy as np 
import matplotlib.pyplot as plt   
x = np.array ([1,2,3,4,5])
y = np.array ([3,4,5])

xx,yy = np.meshgrid(x,y)
print(xx)
print(yy)

print(xx.ravel())
print(yy.ravel())   
print(np.array([xx.ravel(),yy.ravel()]))
print(np.array([xx.ravel(),yy.ravel()]).T)