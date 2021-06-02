import numpy
import additional
import copy
from numpy.linalg import norm
import task1

def TransformToEqual(size, a, b):
    alpha = [[0.0] * size for i in range(size)]
    beta = [0.0 for i in range(size)]

    swaps = []
    for i in range(size):
        if a[i][i] == 0:
            for j in range(size):
                if a[j][i] != 0 and a[i][j] != 0:
                    swaps.append((i, j))
                    a[i], a[j] = a[j], a[i]

    for i in range(size):
        beta[i] = b[i] / a[i][i]
        for j in range(size):
            alpha[i][j] = -a[i][j] / a[i][i]
        alpha[i][i] = 0

    return alpha, beta

def SimpleIteration(size, a, b, precision):
    alpha, beta = TransformToEqual(size, a, b)
    alphaNorm = additional.MatrixNorm(alpha)
    x = beta.copy()
    i = 0
    while True:
        prev = x.copy()
        multAlfaX = additional.MultiplyMatrivOnVector(size, alpha, x)
        x = additional.SummVectors(size, beta, multAlfaX)
        norm = additional.VectorNorm(size, additional.SubtractVectors(size, x, prev))
        print("Ответ на %i-ой итерации" % i)
        i += 1
        additional.PrintVector(x)
        if alphaNorm >= 1:
            if norm * alphaNorm / (1 - alphaNorm) <= precision: # если alphaNorm >= 1, norm <= precision
                break
        elif norm <= precision:
            break
    return x

def SimpleIterationMethod(size, a, b, precision):
    print("Метод итераций")
    x = SimpleIteration(size, copy.deepcopy(a), copy.deepcopy(b), precision)
    print("Ответ с точностю", precision, "полученный методом простых итераций")
    additional.PrintVector(x)

def Seidel1(size, a, b):
    alpha, beta = TransformToEqual(size, a, b)
    b = copy.deepcopy(alpha)
    for i in range(size):
        for j in range(i, size):
            b[i][j] = 0
    k = copy.deepcopy(alpha)
    for i in range(size):
        for j in range(i):
            k[i][j] = 0
    print()

def Seidel(size, a, b, precision):
    print("Метод Зейделя")
    Seidel1(size, a, b)
    alpha, beta = TransformToEqual(size, a, b)

    alpha = numpy.array(alpha)
    beta = numpy.array(beta)
    # нижняя треугольная матрица с нулевой диагональю
    b = copy.deepcopy(alpha)
    for i in range(size):
        for j in range(i, size):
            b[i][j] = 0
    # верхняя треугольная матрица
    c = copy.deepcopy(alpha)
    for i in range(size):
        for j in range(i):
            c[i][j] = 0

    reverseB = task1.InverseMatrix(numpy.eye(size, size) - b) # обратную матрицу из своей лабораторной 1
    t1 = reverseB @ c
    t2 = reverseB @ beta
    x = t2
    cNorm = additional.NPNorm(a, alpha, t1, c)
    i = 0
    while True:
        print("Ответ на %i-ой итерации" % i)
        i += 1
        curX = t2 + t1 @  x
        additional.PrintVector(curX)
        if cNorm * norm(curX - x, numpy.inf) <= precision:
            break
        x = curX
    x = curX
    return x

def SeidelMethod(size, a, b, precision):
    x = Seidel(size, a, b, precision)
    print("Ответ с точностью", precision, "полученный методом Зейделя")
    additional.PrintVector(x)
    x = numpy.linalg.solve(a, b)
    print("Ответ, почлученный через numpy.linalg")
    additional.PrintVector(x)

if __name__ == '__main__':
    size, a, b, precision = additional.GetStatement("3.txt")
    print("Лабораторная работа 1.3")
    print()
    additional.PrintMatrix(a)
    additional.PrintVector(b)
    SimpleIterationMethod(size, a, b, precision)
    SeidelMethod(size, a, b, precision)
