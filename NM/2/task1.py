import math
import numpy
import matplotlib.pyplot as plot
import subprocess, sys

def DrawChart(function, functionDerivative, n, fileName, secondFunctionDerivative, step=0.5, label1='Функция', label2='Её производная', label3='Её вторая производная'):
    range = numpy.arange(n[0], n[-1], step)
    functionValues = [function(i) for i in range]
    functionDerivativeValues = [functionDerivative(i) for i in range]

    figure, ax = plot.subplots()
    ax.plot(range, functionValues, label=label1)
    ax.plot(range, functionDerivativeValues, label=label2)

    secondFunctionDerivativeValues = [secondFunctionDerivative(i) for i in range]
    ax.plot(range, secondFunctionDerivativeValues, label=label3)

    ax.legend(loc='upper right')
    ax.grid()

    figure.savefig(fileName)
    plot.close(figure)
    plot.show()
    opener = "open" if sys.platform == "darwin" else "xdg-open"
    subprocess.call([opener, fileName])

def Fx(x):
    return x

def Fi(x):
    return math.sqrt((math.sin(x) + 0.5) / 2)

def FiDerivative(x):
    return math.sqrt(abs(-math.cos(x) / 2))

def SimpleIterationMethod(Fi, FiDerivative, begin, end, precision=0.001, fileName = 'SimpleIterationMethod.png'):
    print(f'Метод простых итераций c точностью {precision}:')
    DrawChart(Fi, Fx, [0, 1.5], fileName, step=0.1, secondFunctionDerivative=FiDerivative, label2='φ', label3='Производная φ')
    q = max(abs(FiDerivative(begin)), abs(FiDerivative(end)))
    x = (end + begin) / 2
    xi = x
    i = 0
    while True:
        i += 1
        xi = Fi(x)

        print(f'Ответ {xi} на {i}-ой итерации с точностью {abs(xi - x)}.')
        if q / (1 - q) * abs(xi - x) <= precision:
            print()
            return xi
        x = xi

def Function(x):
    return math.sin(x) - 2 * x ** 2 + 0.5

def FunctionDerivative(x):
    return math.cos(x) - 4 * x

def SeconFunctionDerivative(x):
    return -math.sin(x) - 4

def NewtonMethod(function, functionDerivativeValues, xi, precision=0.001, fileName='NewtonMethod.png'):
    print(f'Метод Ньютона c точностью {precision}:')
    DrawChart(Function, FunctionDerivative, [-2, 4], fileName, step=0.1, secondFunctionDerivative=SeconFunctionDerivative)
    x = xi
    i = 0
    while True:
        i += 1
        xi = x - function(x) / functionDerivativeValues(x)
        print(f'Ответ {xi} на {i}-ой итерации с точностью {abs(xi - x)}.')
        if abs(xi - x) <= precision:
            print()
            return xi
        x = xi

if __name__ == '__main__':
    SimpleIterationMethod(Fi, FiDerivative, 0.75, 1)
    NewtonMethod(Function, FunctionDerivative, 0.75)
