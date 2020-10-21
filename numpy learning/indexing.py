import numpy as np  
# for one dimensional array
'''a = np.arange(10)**2
print(a)
print(a[2:5])

a[:6:2] = 1000
print(a)

for i in a:
    print(i**(1/3))'''
    
#multidimensional array
def f(x,y):
    return 10*(x+y)

a = np.fromfunction(f,(5,4),dtype= int)
print(a)

print(a[2,3])
print(a[1:3,1:3])

# : means each
print(a[:,:])
print(a[:,1:3])