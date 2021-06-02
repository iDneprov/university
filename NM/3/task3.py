import numpy as np
import matplotlib.pyplot as plt
import subprocess, sys
import lab1task1

statement = {"x1": [-5, -3, 1, 1, 3, 5], "y1": [-1.3734, -1.249, -0.7854, 0.7854, 1.249, 1.3734], "x": [0.0, 1.7, 3.4, 5.1, 6.8, 8.5], "y": [0.0, 1.3038, 1.8439, 2.2583, 2.6077, 2.9155]}

def GetDegreePolynom(x, y, n):
    equation = [len(x)] + [0] * (2 * n)
    systemRight = [0] * (n + 1)

    for i in range(len(x)):
        for k in range(1, 2 * n + 1):
            equation[k] += x[i] ** k
        for k in range((n + 1)):
            systemRight[k] += y[i] * (x[i] ** k)

    shift = 0
    systemLeft = []
    for j in range(n + 1):
        systemLeft.append([equation[i + shift] for i in range(n + 1)])
        shift += 1

    a = lab1task1.LUP(systemLeft, systemRight)

    def f(x):
        return a[0] + sum(a[i] * (x ** i) for i in range(1, len(a)))

    return a, f

def CalculateSqareErrorsSumm(x, y, f):
    return sum((f(x[i]) - y[i]) ** 2 for i in range(len(x)))

def PrintPolynom(x, f):
    for i in range(len(x)):
        print("{:7.4f}".format(x[i]), end=" | ")
    print()
    y = [f(x[i]) for i in range(len(x))]
    for i in range(len(y)):
        print("{:7.4f}".format(y[i]), end=" | ")
    print()

def DrawChart(f, fileName, step=0.1):
    x = np.arange(statement["x"][0], statement["x"][-1], step)
    y = []
    for i in range(len(f)):
        y.append([f[i](val) for val in x])

    fig, ax = plt.subplots()
    for i in range(len(y)):
        ax.plot(x, y[i], label=f'приближ мн-член {i + 1}-ой степени')

    ax.plot(statement["x"],  statement["y"], 'ro')
    ax.legend(loc='upper left')

    ax.grid()

    if fileName:
        fig.savefig(fileName)
        print(f"Saved in {fileName} succesfuly")
        plt.close(fig)
        opener = "open" if sys.platform == "darwin" else "xdg-open"
        subprocess.call([opener, fileName])

    plt.show()

def LeastSqaresMethod():
    x = statement["x"]
    y = statement["y"]
    polynoms = []
    for degree in range(1, 5):
        print(f"Приближающего многочлена {degree}-ой степени:")
        a, polynom = GetDegreePolynom(x, y, degree)
        polynoms.append(polynom)
        PrintPolynom(x, polynom)
        sqareErrorsSumm = CalculateSqareErrorsSumm(x, y, polynom)
        print("Сумма квадратов ошибок: {:6.6f}\n".format(sqareErrorsSumm))
    DrawChart(polynoms, "LeastSqaresMethod.png")

if __name__ == '__main__':
    LeastSqaresMethod()
