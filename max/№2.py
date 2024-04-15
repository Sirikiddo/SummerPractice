from random import random
import matplotlib.pyplot as plt
import numpy as np
import math


class CreateField():
    def __init__(self, height, width, rho):
        self.height = height
        self.width = width
        self.rho = rho
        self.d = 1/((rho)**(1/2))

    def Data(self):
        self.xp = []
        self.yp = []
        self.n = math.ceil(self.height*self.width*self.rho)
        for i in range(self.n):
            self.xp.append(self.width * random() - self.width/2)
            self.yp.append(self.height * random() - self.height/2)

    def Grid(self):
        NumOfPointsG = int(self.height / self.d)
        GPoints = []
        xg,yg = np.linspace(math.floor(-self.width/2), math.ceil(self.width/2), NumOfPointsG), np.zeros(NumOfPointsG)
        for i in range(NumOfPointsG + 1):
            GPoints.append(-self.height/2 + i*self.d) 
        for i in range(NumOfPointsG + 1):
            plt.plot(xg,yg + GPoints[i],color = 'black',linewidth = 0.5)

        NumOfPointsV = int(self.width / self.d)
        VPoints = []
        xv,yv = np.zeros(NumOfPointsV),np.linspace(math.floor(-self.height/2), math.ceil(self.height/2), NumOfPointsV)
        for i in range(NumOfPointsV + 1):
            VPoints.append(-self.width/2 + i*self.d)
        for i in range(NumOfPointsV + 1):
            plt.plot(xv + VPoints[i],yv,color = 'black',linewidth = 0.5)
            
    def CreatePlot(self,xBound1,xBound2,yBound1,yBound2):
        self.Data()
        plt.scatter(self.xp,self.yp)
        plt.axis('equal')
        plt.xlim(xBound1,xBound2)
        plt.ylim(yBound1,yBound2)
        plt.xlabel('x')
        plt.ylabel('y')
        self.Grid()
        plt.show()

a = CreateField(10,20,1)
a.CreatePlot(0.5,1.5,3.5,4.5)
