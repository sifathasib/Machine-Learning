import numpy as np

class Perceptron(object):
    def __init__(self,eta = 0.01,n_itr = 50,random_state=1):
        self.eta = eta
        self.n_itr = n_itr
        self.random_state = random_state
        
    def fit(self,X,y):
        rgen = np.random.RandomState(self.random_state)
        self.weight = rgen.normal(loc=0,scale=0.1,size=1+X.shape[1])
        self.errors_ = []
        for _ in range(self.n_itr):
            error =0
            for xi ,target in zip(X,y):
                update = self.eta*(target-self.predict(xi))
                self.weight[1:] = update*xi
                self.weight[0] = update
                error += int(update!= 0.0)
            self.errors_.append(error)
        return self
    def net_input(self,xi):
        return np.dot(xi,self.weight[1:])+self.weight[0]
    def predict(self,xi):
        return np.where(self.net_input(xi)>= 0.0,1,-1)