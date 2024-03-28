import random
import math
import matplotlib.pyplot as plt


class GenRandPoint():
    def __init__(self, height, width, rho):
        self.height = height
        self.width = width
        self.rho = rho

    def gen(self):
        ar_x = []
        ar_y = []
        n = math.ceil(self.height * self.width * self.rho)
        for i in range(0, n):
            x = self.width * random.random() - self.width / 2
            y = self.height * random.random() - self.height / 2
            ar_x.append(x)
            ar_y.append(y)
        return ar_x, ar_y

    def draw(self):
        generation = self.gen()
        plt.scatter(generation[0], generation[1])
        plt.xlim([0.5, 1.5])
        plt.ylim([3.5, 4.5])
        plt.show()


print(123)
z = GenRandPoint(8, 8, 10)
GenRandPoint.draw(self=z)
