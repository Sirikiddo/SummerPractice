import random
import math
import matplotlib.pyplot as plt


class GenRandPoint():
    def __init__(self, height, width, rho):
        self.height = height
        self.width = width
        self.rho = rho
        self.d = 1 / (rho ** 0.5)

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
        d = self.d
        generation = self.gen()
        plt.scatter(generation[0], generation[1])

        # Рисуем вертикальные линии сетки
        for i in range(-int(self.width / (2 * d)), int(self.width / (2 * d)) + 1):
            plt.axvline(x=i * d, color='gray', linestyle='--', linewidth=0.5)

        # Рисуем горизонтальные линии сетки
        for i in range(-int(self.height / (2 * d)), int(self.height / (2 * d)) + 1):
            plt.axhline(y=i * d, color='gray', linestyle='--', linewidth=0.5)

        plt.xlim([0.5, 1.5])
        plt.ylim([3.5, 4.5])
        plt.show()


z = GenRandPoint(8, 8, 10)
GenRandPoint.draw(self=z)
