import math

def Rastrigin(x, A=2):
    dim = len(x)
    f = A * dim
    for element in x:
        f += element ** 2 - A * math.cos(2 * math.pi * element)
    return f


def Rosenbrock(x):
    dim = len(x)
    f = 0
    for i in range((len(x) - 1)):
        f += 100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2
    return f


def Eggholder(xbar):
    x = xbar[0]
    y = xbar[1]
    return - (y + 47) * math.sin(math.sqrt(abs(x / 2 + y + 47))) - x * math.sin(math.sqrt(abs(x - y - 47)))


def Ackley(xbar):
    x = xbar[0]
    y = xbar[1]
    return  -20 * math.e ** (-0.2 * math.sqrt(0.5 * (x ** 2 + y ** 2))) - math.e ** (0.5 * (math.cos(2 * math.pi * x) + math.cos(2 * math.pi * y))) + math.e + 20
