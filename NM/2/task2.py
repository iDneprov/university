import math
import numpy as np
import matplotlib.pyplot as plt
import subprocess, sys

def DrawChart(FirstFunction, SecondFunction, xNew, fileName, step=0.1):
    xNew = np.arange(xNew[0], xNew[-1], step)
    y1 = [FirstFunction(i) for i in xNew]
    y2 = [SecondFunction(i) for i in xNew]

    fig, ax = plt.subplots()
    ax.plot(xNew, y1, label='Первая функция')
    ax.plot(y2, xNew, label='Вторая функция')

    ax.legend(loc='upper right')
    ax.grid()

    fig.savefig(fileName)
    plt.close(fig)

    plt.show()
    opener = "open" if sys.platform == "darwin" else "xdg-open"
    subprocess.call([opener, fileName])


def Fi1(XFromFirstFunction, XFromSecondFunction):
    return math.cos(XFromSecondFunction) + 1

def Fi1DX1(XFromFirstFunction, XFromSecondFunction):
    return 0

def Fi1DX2(XFromSecondFunction):
    return -math.sin(XFromSecondFunction)


def Fi2(XFromFirstFunction, XFromSecondFunction):
    return math.sin(XFromFirstFunction) + 1

def Fi2DX1(XFromFirstFunction):
    return math.cos(XFromFirstFunction)

def Fi2DX2(XFromFirstFunction, XFromSecondFunction):
    return 0

def QCalculate(XFromFirstFunction, XFromSecondFunction):
    return max(
        abs(Fi1DX1(XFromFirstFunction, XFromSecondFunction)) + abs(Fi1DX2(XFromSecondFunction)),
        abs(Fi2DX1(XFromFirstFunction)) + abs(Fi2DX2(XFromFirstFunction, XFromSecondFunction)))

def SimpleIterationMethod(x1, x2, precision=0.01, fileName = 'SimpleIterationMethod.png'):
    print(f'Метод простых итераций с точностью {precision}:')
    DrawChart(Fi1DX2, Fi2DX1, [-3, 5], fileName)
    x1New = x1
    x2New = x2
    i = 0
    while True:
        i += 1
        x1New = Fi1(x1, x2)
        x2New = Fi2(x1, x2)
        q = QCalculate(x1, x2)
        if q >= 1:
            print('Выберите другой интервал!')
            return
        p = abs((q / (1 - q)) * ErrorSсale((x1New, x2New), (x1, x2)))
        print(f'На {i}-ой итерации x1 = {x1New}, x2 = {x2New} с точностю {p}.')
        if p <= precision:
            break
        x1 = x1New
        x2 = x2New
    return x1New, x2New


def FirstFunction(XFromFirstFunction, XFromSecondFunction):
    return XFromFirstFunction - math.cos(XFromSecondFunction) - 1

def XFromFirstFunction(XFromSecondFunction):
    return math.cos(XFromSecondFunction) + 1

def FirstFunctionDX1(XFromFirstFunction, XFromSecondFunction):
    return 1

def FirstFunctionDX2(XFromFirstFunction, XFromSecondFunction):
    return math.sin(XFromSecondFunction)


def SecondFunction(XFromFirstFunction, XFromSecondFunction):
    return XFromSecondFunction - math.sin(XFromFirstFunction) - 1

def XFromSecondFunction(XFromFirstFunction):
    return math.sin(XFromFirstFunction) + 1

def SecondFunctionDX1(XFromFirstFunction, XFromSecondFunction):
    return -math.cos(XFromFirstFunction)

def SecondFunctionDX2(XFromFirstFunction, XFromSecondFunction):
    return 1

def Determinant(a):
    return a[0][0] * a[1][1] - a[0][1] * a[1][0]

def NewX(XFromFirstFunction, XFromSecondFunction):
    a1 = [
        [FirstFunction(XFromFirstFunction, XFromSecondFunction), FirstFunctionDX2(XFromFirstFunction, XFromSecondFunction)],
        [SecondFunction(XFromFirstFunction, XFromSecondFunction), SecondFunctionDX2(XFromFirstFunction, XFromSecondFunction)]
    ]
    a2 = [
        [FirstFunctionDX1(XFromFirstFunction, XFromSecondFunction), FirstFunction(XFromFirstFunction, XFromSecondFunction)],
        [SecondFunctionDX1(XFromFirstFunction, XFromSecondFunction), SecondFunction(XFromFirstFunction, XFromSecondFunction)]
    ]

    j = [
        [FirstFunctionDX1(XFromFirstFunction, XFromSecondFunction), FirstFunctionDX2(XFromFirstFunction, XFromSecondFunction)],
        [SecondFunctionDX1(XFromFirstFunction, XFromSecondFunction), SecondFunctionDX2(XFromFirstFunction, XFromSecondFunction)]
    ]
    DeterminantJ = Determinant(j)
    return XFromFirstFunction - Determinant(a1) / DeterminantJ, XFromSecondFunction - Determinant(a2) / DeterminantJ

def ErrorSсale(xNew, x):
    return max(xNew[0] - x[0], xNew[1] - x[1])

def NewtonMethod(x1, x2, precision=0.01, fileName='NewtonMethod.png'):
    print(f'Метод Ньютона с точностью {precision}:')
    DrawChart(XFromFirstFunction, XFromSecondFunction, [-3, 5], fileName)
    x1New = x1
    x2New = x2
    i = 0
    while True:
        i += 1
        x1New, x2New = NewX(x1, x2)
        p = ErrorSсale((x1New, x2New), (x1, x2))
        print(f'На {i}-ой итерации x1 = {x1New}, x2 = {x2New} с точностю {p}.')
        if p <= precision:
            break
        x1 = x1New
        x2 = x2New
    return x1New, x2New


if __name__ == '__main__':
    NewtonMethod(x1=0.5, x2=2, precision=0.001)
    SimpleIterationMethod(x1=0.5, x2=2, precision=0.0001)
