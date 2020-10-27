import numpy as np 
a = np.array([[1,2],
              [3,4],
              [5,6]])
print(a)
point = np.array([1,2])
def testing(x):
    return x*2

print(testing(point))
z = testing(a)
print(z)