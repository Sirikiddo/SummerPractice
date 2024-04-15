import matplotlib.pyplot as plt
import scipy.stats.qmc as qmc
import numpy as np
import math

class CreateField():
    def __init__(self, height, width, rho):
        self.height = height
        self.width = width
        self.rho = rho
        self.n = math.ceil(self.height*self.width*self.rho)
        
    def Data(self,method):
        data = method()
        self.xp = data[:,0]
        self.yp = data[:,1]
        for i in range(0,self.n):
            self.xp[i] = (self.width * self.xp[i] - self.width/2)
            self.yp[i] = (self.height * self.yp[i] - self.height/2)

    def Sobol(self):
        S = qmc.Sobol(2)
        return S.random(self.n)
        
    def Halton(self):
        H = qmc.Halton(2)
        return H.random(self.n)
        
    def CreatePlot(self,method,xBound1,xBound2,yBound1,yBound2):
        self.Data(method)
        plt.scatter(self.xp,self.yp)
        plt.axis('equal')
        plt.xlim(xBound1,xBound2)
        plt.ylim(yBound1,yBound2)
        plt.title(f'{method.__name__}')
        plt.xlabel('x')
        plt.ylabel('y')

a = CreateField(10,20,5)
fig, axes = plt.subplots(2,1, figsize=(8, 8))
plt.subplot(2,1,1)
a.CreatePlot(a.Halton,-11,11,-6,6)
plt.subplot(2,1,2)
a.CreatePlot(a.Sobol,-11,11,-6,6)
plt.tight_layout()
plt.show()
