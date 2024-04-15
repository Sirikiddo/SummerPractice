from random import random
import matplotlib.pyplot as plt
import numpy as np
import math

height = 20
width = 10
centerPoint = (0,0)
rho = 2

xp = []
yp = []

n = math.ceil(height*width*rho)
for i in range(n):
    xp.append(width * random() - width/2)
    yp.append(height * random() - height/2)


fig,ax = plt.subplots()
ax.scatter(xp,yp, s=30, facecolor='C0', edgecolor='k')
plt.grid(True)
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')
plt.xlim(0.5,1.5)
plt.ylim(3.5,4.5)
fig.show()
