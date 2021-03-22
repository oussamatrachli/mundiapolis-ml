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