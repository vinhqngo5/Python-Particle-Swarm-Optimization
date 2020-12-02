import matplotlib.pyplot as plt
import math
plt.style.use('seaborn-white')
import numpy as np

def Rastrigin_contour(x, y, A=10):
    a = x ** 2 - A * np.cos(2 * math.pi * x)
    b = y ** 2 - A * np.cos(2 * math.pi * y)
    return a + b

def Rosenbrock_contour(x, y):
    return 100 * (y - x ** 2) ** 2 + (1 - x) ** 2

def Eggholder_contour(x, y):
    return - (y + 47) * np.sin(np.sqrt(np.abs(x / 2 + y + 47))) - x * np.sin(np.sqrt(np.abs(x / 2 - y - 47)))

def Ackley_contour(x, y):
    return  -20 * math.e ** (-0.2 * np.sqrt(0.5 * (x ** 2 + y ** 2))) - math.e ** (0.5 * (np.cos(2 * math.pi * x) + np.cos(2 * math.pi * y))) + math.e + 20


def draw_contour(func, lb, ub, des='/content/pyswarm/plot.png', num=0, a1=[], a2=[], a3=[]):
    fig, ax = plt.subplots()
    x = np.linspace(lb, ub, 50)
    y = np.linspace(lb, ub, 40)
    X, Y = np.meshgrid(x, y)
    Z = func(X, Y)
    # cs = ax.contourf(X, Y, Z)
    cs = ax.contourf(X, Y, Z, 10, cmap = 'bone')
    cbar = plt.colorbar(cs) 
    ax.plot(a1, a2, 'ro', markersize=5)
    fig.suptitle('Generation No.{}'.format(num), fontsize=12)
    plt.savefig(des, dpi=140, bbox_inches='tight')
    return plt



