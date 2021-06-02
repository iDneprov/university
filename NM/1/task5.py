import numpy
import math
import additional
import task1
import copy

def QRDecompose(size, aOriginal):
    a = copy.deepcopy(aOriginal)
    e = [[0 if i != j else 1 for i in range(size)] for j in range(size)] # единичная матрица
    q = copy.deepcopy(e)
    # находим векторы v и v транспанированный
    for i in range(size):
        v = [0.0 for i in range(size)]
        v[i] = a[i][i] + additional.Sign(a[i][i]) * math.sqrt(sum(a[j][i] ** 2 for j in range(i, size)))
        for j in range(i + 1, size):
            v[j] = a[j][i]
        vTransposed = [v]
        v = [[v[i]] for i in range(size)]
        # получаем матрицу h
        dividend = additional.MultiplyMatrix(v, vTransposed)
        divider = additional.MultiplyMatrix(vTransposed, v)[0][0]
        for k in range(size):
            for y in range(size):
                dividend[k][y] = 2 * dividend[k][y] / divider
        h = additional.SubstractMaxrix(e, dividend)
        # получаем матрицы q и a для текущего шага
        q = additional.MultiplyMatrix(q, h)
        a = additional.MultiplyMatrix(h, a)

    return q, a

def Solve(a, rootType):
    n = len(a)
    a = numpy.array(a)
    soLUPtion = []
    k = 0
    for t in rootType:
        if t == 'real':
            soLUPtion.append(a[k, k])
        else:

            A11 = a[k, k]
            A12 = A21 = A22 = 0

            if k + 1 < n:
                A12 = a[k, k + 1]
                A21 = a[k + 1, k]
                A22 = a[k + 1, k + 1]

            soLUPtion.extend(numpy.roots(
                (1, -A11 - A22, A11 * A22 - A12 * A21)))
            k += 1
        k += 1
    return soLUPtion

def GetRootTipe(size, matrix, precision):
    rootType = []
    k = 0
    while k < size:
        if additional.SNorm(additional.GetVectorOfCoeffs(matrix, k)) <= precision:
            rootType.append('real')
        elif additional.SNorm(additional.GetVectorOfCoeffs(matrix, k + 1)) <= precision:
            rootType.append('comlex')
            k += 1
        else:
            rootType.append('not root')
        k += 1
    return rootType

def QRMain(size, aOriginal, precision):
    a = copy.deepcopy(aOriginal)
    step = True
    for i in range(101):
        q, r = QRDecompose(size, a)
        a = additional.MultiplyMatrix(r, q)
        print("Матрица на %i-ой итерации" % i)
        additional.PrintMatrix(q)

        rootType = GetRootTipe(size, a, precision)
        if additional.HaveNoRoot(rootType):
            if step:
                step = False
            else:
                return Solve(a, rootType)

    return None


def QRMethod(size, a, precision):
    print("Лабораторная работа 1.5")
    print("Алгоритм QR – разложения матриц")
    print()
    print("Матрица")
    additional.PrintMatrix(a)
    x = QRMain(size, a, precision)
    print("Ответ")
    additional.PrintVector(x)

    x, u = numpy.linalg.eig(a)
    print("Ответ, почлученный через numpy.linalg")
    additional.PrintVector(x)

if __name__ == '__main__':
    size, a, b, precision = additional.GetStatement("55.txt")
    QRMethod(size, a, precision)
