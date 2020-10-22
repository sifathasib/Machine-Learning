import numpy as np

'''a = np.arange(12)**2
i = np.array([1,1,2,4,5,7])
j= np.array([[3,4],[5,7]])
print(a)
print(a[i])
print(a[j])'''

'''pallate = np.array([[0,0,0],
                    [0,255,0],
                    [255,0,0],
                    [0,0,255],
                    [255,255,255]])

image = np.array([[0,1,2,0],
                  [0,3,4,0]])

print(pallate[image])'''

a = np.arange(12).reshape(3,4)
print(a)
i = np.array([[1,1],
              [2,2]])
j = np.array([[0,2],
              [1,3]])

print(a[i,j])
