import numpy as numpy
import additional
import algo
from pprint import pprint

def LUPMethod(size, a, b):
    print("Лабораторная работа 1.1")
    print("алгоритм LUP - разложения матриц ")
    print()
    l, u, p = algo.LUPSeparate(size, a)
    x = algo.LUPSolve(size, l, u, b, p)
    print("Матрица")
    additional.PrintMatrix(a)
    additional.PrintVector(b)
    print("L")
    additional.PrintMatrix(l)
    print("U")
    additional.PrintMatrix(u)
    print("Ответ")
    additional.PrintVector(x)

    nL = numpy.array(l)
    nU = numpy.array(u)
    print("Ответ, почлученный через numpy.linalg")
    additional.PrintVector(numpy.linalg.solve(a, b))

def RunThroughMethod(size, a, b):
    print("Лабораторная работа 1.2")
    print("Метод прогонки")
    print()
    print("Матрица")
    print("A")
    additional.PrintMatrix(a)
    print("B")
    additional.PrintVector(b)
    x = algo.tridiagonal(size, a, b)
    print("Ответ")
    additional.PrintVector(x)
    print("Ответ, почлученный через numpy.linalg")
    additional.PrintVector(numpy.linalg.solve(a, b))

def SimpleIterationMethod(size, a, b, precision):
    print("Метод простых итераций")
    print()
    print("Матрица")
    print("A")
    additional.PrintMatrix(a)
    print("B")
    additional.PrintVector(b)
    x = algo.SimpleIteration(size, a, b, precision)
    print("Ответ с точностю", precision)
    additional.PrintVector(x)
    print("Ответ, почлученный через numpy.linalg")
    additional.PrintVector(numpy.linalg.solve(a, b))

def SeidelMethod(size, a, b, precision):
    print("Метод Зейделя")
    print()
    print("Матрица")
    additional.PrintMatrix(a)
    additional.PrintVector(b)
    x = algo.zeidel_method(size, a, b, precision)
    print("Ответ")
    additional.PrintVector(x)
    print("Ответ, почлученный через numpy.linalg")
    additional.PrintVector(numpy.linalg.solve(a, b))

def Task3(size, a, b, precision):
    print("Лабораторная работа 1.3")
    print()
    SimpleIterationMethod(size, a, b, precision)
    SeidelMethod (size, a, b, precision)


def RotationMethod(size, a, b, precision):
    print("Лабораторная работа 1.4")
    print("Метод вращений")
    print()
    print("Матрица")
    additional.PrintMatrix(a)
    x, u = algo.rotate_jacobi(size, a, precision)
    print("Ответ")
    additional.PrintVector(x)
    additional.PrintMatrix(u)
    print("Ответ, почлученный через numpy.linalg")
    x, u = numpy.linalg.eig(a)
    additional.PrintVector(x)
    additional.PrintMatrix(u)


def QRMethod(size, a, b, precision):
    print("Лабораторная работа 1.5")
    print("Алгоритм QR – разложения матриц")
    print()
    print("Матрица")
    additional.PrintMatrix(a)
    x = algo.QR_method(size, a, precision)
    print("Ответ")
    pprint(x)
    x, u = numpy.linalg.eig(a)
    print("Ответ, почлученный через numpy.linalg")
    pprint(x)

if __name__ == '__main__':
    size, a, b, precision = additional.GetStatement("3.txt")
    SimpleIterationMethod(size, a, b, precision)
