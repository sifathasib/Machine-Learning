from perceptron import Perceptron
import numpy as np  
a = np.array([[1,2],
              [3,4],
              [4,5]])

print(a)
ppn = Perceptron(eta = 0.1,n_itr=10)
z = ppn.predict(a)
print(z)
