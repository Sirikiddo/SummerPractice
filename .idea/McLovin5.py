import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from math import sqrt


def p1(r, f):
    beta_f = (0.1 * f ** 2) / (1 + f ** 2) + (40 * f ** 2) / (4100 + f ** 2) + 2.75 * 10 ** (-4) * f ** 2 + 0.0003
    x = sqrt(f / 10) * ((6.71 * 10 ** 3) / r) * (10 ** (-0.05 * beta_f * 10 ** (-3) * r))
    return erf(x)

def p2(r, f):
    gamma = (f / r ** 2) * 10 ** (0.1 * (SNR(r, f) - beta(f) * 10 ** (-3) * r))
    qe = 0.5 * (1 - sqrt(gamma / (1 + gamma)))
    return (1 - qe) ** 256

def beta(f):
    return (0.1 * f ** 2) / (1 + f ** 2) + (40 * f ** 2) / (4100 + f ** 2) + 2.75e-4 * f ** 2 + 0.0003

def SNR(r, f):
    return 0

# Создание массивов значений для r и f
r_values = np.arange(100, 2001, 200)  # от 100 до 2000 с шагом 200
f_values = np.arange(20, 101, 10)  # от 20 до 100 с шагом 10

# Построение графиков для p1 и p2 по r при изменяющемся f
plt.figure(figsize=(12, 6))
for f in f_values:
    p1_values = [p1(r, f) for r in r_values]
    p2_values = [p2(r, f) for r in r_values]
    plt.plot(r_values, p1_values, label=f'p1, f={f} kHz')
    plt.plot(r_values, p2_values, label=f'p2, f={f} kHz', linestyle='dashed')
plt.xlabel('Расстояние между сенсорами, м')
plt.ylabel('Вероятность')
plt.title('Графики зависимости вероятности от расстояния между сенсорами при разных частотах')
plt.legend()
plt.grid(True)
plt.show()

# Построение графиков для p1 и p2 по f при изменяющемся r
plt.figure(figsize=(12, 6))
for r in r_values:
    p1_values = [p1(r, f) for f in f_values]
    p2_values = [p2(r, f) for f in f_values]
    plt.plot(f_values, p1_values, label=f'p1, r={r} m')
    plt.plot(f_values, p2_values, label=f'p2, r={r} m', linestyle='dashed')
plt.xlabel('Частота передачи данных, кГц')
plt.ylabel('Вероятность')
plt.title('Графики зависимости вероятности от частоты передачи данных при разных расстояниях')
plt.legend()
plt.grid(True)
plt.show()
