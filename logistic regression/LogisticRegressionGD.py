import numpy as np
class LogisticRegressionGd(object):
    
    def __init__(self, eta= 0.1,n_itr= 100,random_state=1):
        self.eta = eta
        self.n_itr = n_itr
        self.random_state = random_state
    
    def fit(self,X,y):
        rgen = np.random.RandomState(self.random_state)
        self.weight = rgen.normal(loc=0, scale=0.01,size=1+X.shape[1])
        self.cost_=[]
        
        for i in range(self.n_itr):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y-output)    
            self.weight[1:] += self.eta * np.dot(X.T,errors)
            self.weight[0] += self.eta*errors.sum()
            cost = -y.dot(np.log(output))-((1-y).dot(np.log(1-output)))
            self.cost_.append(cost)
        return self   
    def net_input(self,X):
        return np.dot(X,self.weight[1:])+self.weight[0]
    
    def activation(self,z):
        return 1/(1+np.exp(np.clip(z,-250,250)))
    def predict(self,X):
        return np.where(self.activation(self.net_input(X))>=0.5,1,0)