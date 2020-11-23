import matplotlib.pyplot as plt 
import numpy as np 

def gini(p):
    return (p)*(1-(p))+(1-p)*(1-(1-p))

def entropy(p):
    return -p*np.log2(p)-(1-p)*np.log2((1-p))
def error(p):
    return 1 - np.max([p,1-p])

x = np.arange(0,1,0.01)

entropy=[entropy(p) if p!=0 else None for p in x]
sc_entropy = [e*0.5 if e else None for e in entropy]    
err = [error(i) for i in x]
fig = plt.figure()
ax = plt.subplot(111)

for i,lab,ls,c in zip([entropy,sc_entropy,gini(x),err],['Entropy','Scaled Entropy','Gini Impurity','Misclassification error'],['-','-','--','-.'],['black','red','green','cyan']):
    line = ax.plot(x,i,label=lab,linestyle=ls,lw=2,c=c)
ax.legend(loc='upper center',bbox_to_anchor=(0.5,1.15),ncol=5,fancybox=True,shadow=False)
ax.axhline(y=0.5,linewidth= 1,color='k',linestyle='--')
ax.axhline(y=1.0,linewidth= 1,color='k',linestyle='--')

plt.ylim([0,1.1])
plt.xlabel('p(i=1)')
plt.ylabel('impurtiy Index')
plt.show()


