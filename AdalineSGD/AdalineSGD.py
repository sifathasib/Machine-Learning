import numpy as np  

class AdalineSGD(object):
    
    def __init__(self,eta=0.1,n_itr= 10,shuffle = True,random_state=None):
        self.eta =eta  
        self.n_itr = n_itr
        self.shuffle = shuffle
        self.random_state = random_state
        self.weight_initialized = False
    def fit(self,X,y):
        self.initialize_weights(X.shape[1])
        self.cost_ =[]
        for i in range(self.n_itr):
            if self.shuffle:
                X,y = self._shuffle(X,y)
            cost = []
            for xi,target in zip(X,y):
                cost.append(self.update_weights(xi,target))
            avg_cost = sum(cost)/len(y)
            self.cost_.append(avg_cost)
        return self  
    def partial_fit(self, X, y):
        if not self.weight_initialized:
            self.initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi,target in zip(X,y):
                self.update_weights(xi,target)
        else:
            self.update_weights(X,y)
        return self 
    def _shuffle(self,X,y):
        r = self.rgen.permutation(len(y))
        return X[r],y[r]
    
    def initialize_weights(self,m):
        self.rgen = np.random.RandomState(self.random_state)
        self.weight = self.rgen.normal(loc = 0.0, scale =0.01,size =1+m)
        self.weight_initialized= True
        
    def update_weights(self,xi,target):
        output = self.activation(self.net_input(xi))
        error = (target - output)
        self.weight[1:] += self.eta * xi.dot(error)
        self.weight[0] += self.eta * error
        cost = 0.5 * error**2
        return cost   
    
    def net_input(self,X):
        return np.dot(X,self.weight[1:]) + self.weight[0]
    def activation(self,X):
        return X
    
    def predict(self,X):
        return np.where(self.activation(self.net_input(X))>= 0.0,1,-1)