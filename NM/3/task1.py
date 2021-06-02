import math
import numpy as np
import matplotlib.pyplot as plt
import subprocess, sys

def Function(x):
    return math.atan(x)

def PolynomValues(polynom, x):
    f = 0
    for i in polynom:
        f += i[0] * MultiplyVectorValuse(x - i[j] for j in range(1, len(i)))
    return f

def MultiplyVectorValuse(gen):
    r = 1
    for i in gen:
        r *= i
    return r

def GetLagrangePolynomPart(i, x):
    lagrangePolynomPart = [1.0]
    for j in range(len(x)):
        if j != i: # нужно для исключения скобки xi - xi
            lagrangePolynomPart.append(x[j])
            lagrangePolynomPart[0] /= x[i] - x[j] # коэффициент, на который умножается скобка
    return lagrangePolynomPart

def GetLagrangePolynom(x, f):
    y = [f(i) for i in x]
    lagrangePolynom = []

    for i in range(len(x)):
        lagrangePolynomPart = GetLagrangePolynomPart(i, x)
        lagrangePolynomPart[0] *= y[i] # коэффициент, на который умножается скобка
        lagrangePolynom.append(lagrangePolynomPart)

    return lagrangePolynom

def GetNewtonPolynom(x, f):
    fValues = [[f(i) for i in x]]
    newtonPolynom = [[fValues[0][0]]]
    K = 1

    for i in range(len(x) - 1):
        fValues.append([])
        for j in range(len(fValues[i]) - 1):
            fValues[i + 1].append((fValues[i][j] - fValues[i][j + 1]) / (x[j] - x[j + K])) # заполняем таблицу конечных разностей
        newtonPolynom.append([fValues[i + 1][0]] + [x[k] for k in range(i + 1)]) # добавляем новый член многочлена
        K += 1

    return newtonPolynom

def PrintPolynom(polynom, type = False):
    if not type:
        print('polynom', end='')
    else:
        print(type + 'Polynom', end='')
    print("(x) = ", end='')
    plus = False
    for i in polynom:
        if i[0] != 0:
            if plus:
                print(" + " if i[0] > 0 else ' ', end='')
            else:
                plus = True
            print("{:5.3f}".format(i[0]), end='')
            for j in range(1, len(i)):
                if i[j] > 0:
                    print(" * (x - {:3.1f})".format(i[j]), end='')
                elif i[j] < 0:
                    print(" * (x + {:3.1f})".format(-i[j]), end='')
                else:
                    print(" * x", end='')
    print()

def LagrangeMethod(Function, testDot, x):
    print("Полином Лагранжа, построенный по точкам", x)
    lagrangePolynom = GetLagrangePolynom(x, Function)
    PrintPolynom(lagrangePolynom, 'Lagrange')
    lagrangePolynomValues = PolynomValues(lagrangePolynom, testDot)
    print("LagrangePolynom({}) = {:6.3f}".format(testDot, lagrangePolynomValues))
    print()
    print("Оценка точности в заданной точке")
    print("|Arctg(x) - LagrangePolynom(x)| = {:6.8f}".format(abs(lagrangePolynomValues - Function(testDot))))
    print()
    print()
    return lagrangePolynom

def NewtonMethod(Function, testDot, x):
    print("Полином Ньютона, построенный по точкам", x)
    newtonPolynom = GetNewtonPolynom(x, Function)
    PrintPolynom(newtonPolynom, 'Newton')
    newtonPolynomValues = PolynomValues(newtonPolynom, testDot)
    print("NewtonPolynom({}) = {:6.3f}".format(testDot, newtonPolynomValues))
    print()
    print("Оценка точности в заданной точке")
    print("|Arctg(x) - NewtonPolynom(x)| = {:6.8f}".format(abs(newtonPolynomValues - Function(testDot))))
    print()
    print()
    return newtonPolynom

def DrawChart(f, x, y, fileName, step=0.15):
    X = np.arange(x[0], x[-1] + step / 2, step)
    Y = []
    for i in range(len(f)):
        Y.append([f[i](val) for val in X])

    fig, ax = plt.subplots()
    for i in range(len(Y)):
        ax.plot(X, Y[i])

    ax.plot(x,  y, 'ro')
    ax.plot(x, y)
    #ax.legend(loc='upper left')

    ax.grid()

    if fileName:
        fig.savefig(fileName)
        plt.close(fig)
        opener = "open" if sys.platform == "darwin" else "xdg-open"
        subprocess.call([opener, fileName])

    plt.show()

def lagrangeValues(x):
    return PolynomValues(lagrangePolynom, x)

def newtonValues(x):
    return PolynomValues(newtonPolynom, x)

if __name__ == '__main__':
    testDot = -0.5
    xA = [-3, -1, 1, 3]
    xB = [-3,  0, 1, 3]

    lagrangePolynom = LagrangeMethod(Function, testDot, xA)
    newtonPolynom = NewtonMethod(Function, testDot, xA)
    DrawChart([lagrangeValues, newtonValues], xA, [Function(x) for x in xA], 'task1A.png')

    lagrangePolynom = LagrangeMethod(Function, testDot, xB)
    newtonPolynom = NewtonMethod(Function, testDot, xB)
    DrawChart([lagrangeValues, newtonValues], xB, [Function(x) for x in xB], 'task1B.png')
