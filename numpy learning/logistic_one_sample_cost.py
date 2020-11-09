import numpy as np   
import matplotlib.pyplot as plt   

def cost_1(z):
    return -np.log(sigmoid(z)) 
def cost_0(z):
    return -np.log(1 - sigmoid(z))
def sigmoid(z):
    return 1/(1+np.exp(-z))

z = np.arange(-10,10,0.1)
phi_z= sigmoid(z)
c1 = [cost_1(x) for x in z]
c0 = [cost_0(x) for x in z]

plt.plot(phi_z,c1,label= "j(w) if y=1")
plt.plot(phi_z,c0,linestyle="--",label= "j(w) if y=0")
plt.ylim(0.0,5.1)
plt.xlim([0,1])
plt.xlabel("$\phi$(z)")
plt.ylabel("j(w)")
plt.legend(loc='best')
plt.show()