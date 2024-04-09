import random
import math
import matplotlib.pyplot as plt
from scipy.stats import qmc


class GenRandPoint():
    def __init__(self, height, width, rho):
        self.height = height
        self.width = width
        self.rho = rho

    def gen(self):
        n = math.ceil(self.height * self.width * self.rho)
        sample = {}

        # Sobol
        engine = qmc.Sobol(2)
        sample["Sobol"] = engine.random(n)

        # Halton
        engine = qmc.Halton(2)
        sample["Halton"] = engine.random(n)

        return sample

    def draw(self):
        sample = self.gen()
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        for i, kind in enumerate(sample):
            axs[i].scatter(sample[kind][:, 0], sample[kind][:, 1])
            axs[i].set_title(f'{kind}')
        plt.tight_layout()
        plt.show()


z = GenRandPoint(8, 8, 4)
GenRandPoint.draw(self=z)
