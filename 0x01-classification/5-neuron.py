import numpy as np

class Neuron:
    def __init__(self,nx):
        if (not(isinstance(nx,int))):
            raise TypeError('nx must be an integer')
        if(nx<1):
            raise ValueError('nx must be a positive integer')
        self.__W = np.random.normal()
        self.__b = 0
        self.__A = 0
    
    def forward_prop(self, X):
        Z = (X[1]*self.__W).sum()+self.__b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A
    
    def cost(self, Y, A):
        p = 1
        counter = 0
        for x in Y:
            if x == 1:
                p = p * np.take(A,[counter])
                counter = counter+1
            else:
                p = p * (1.0000001 - np.take(A,[counter]))
                counter = counter + 1
        cost = (1/counter)*(-np.log10(p))
        return cost

    def evaluate(self, X, Y):
        self.forward_prop(X)
        return np.where(self.A <= 0.5, 0, 1), self.cost(Y, self.A)

    def gradient_descent(self, X, Y, A, alpha=0.05):
        m = Y.shape[1]
        d = A - Y
        gradient = np.matmul(d, X.T) / m
        db = np.sum(d) / m
        self.__W -= gradient * alpha
        self.__b -= db * alpha
