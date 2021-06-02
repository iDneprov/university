import numpy
import additional
import copy

def LUPSeparate(size, a):
    l = [[0.0] * size for i in range(size)]
    j = [[0.0] * size for i in range(size)]
    # делаем p перестановку
    p = []
    u = copy.deepcopy(a)
    for j in range(size - 1):
        tmp = [(u[i][j], i) for i in range(j, size)]
        maxIndex = max(tmp, key=lambda x: abs(x[0]))[1]
        if maxIndex != j:
            p.append((j, maxIndex))
            u[maxIndex], u[j] = u[j].copy(), u[maxIndex].copy()
    # для полученной матрицы делаем LP разбиение
    for i in range(size):
        l[i][i] = 1
        for j in range(i + 1, size):
            l[j][i] = u[j][i] / u[i][i]
            for k in range(i + 1, size):
                u[j][k] -= l[j][i] * u[i][k]

    return l, u, p

def LUPSolve(size, l, u, b, p):
    y = [0 for i in range(size)]
    x = [0 for i in range(size)]

    pMatrix = b.copy()
    # делаем p перестановку
    for i, j in p:
        pMatrix[i], pMatrix[j] = pMatrix[j], pMatrix[i]
    # прямая подстановка
    for i in range(size):
        y[i] = pMatrix[i] - sum(l[i][j] * y[j] for j in range(i))
    # обратная подстановка и нахождение ответа
    for i in reversed(range(size)):
        n = sum(u[i][j] * x[j] for j in range(i + 1, size))
        x[i] = (y[i] - n) / u[i][i]
    return x

def InverseMatrix(a):
    size = len(a)
    l, u, p = LUPSeparate(size, a)
    x = []
    for i in range(size):
        b = [0 if i != j else 1 for j in range(size)]
        x.append(LUPSolve(size, l, u, b, p))
    return x


def LUPMethod(size, a, b):
    print("Лабораторная работа 1.1")
    print("алгоритм LUP - разложения матриц ")
    print()
    print("Матрица")
    additional.PrintMatrix(a)
    additional.PrintVector(b)
    l, u, p = LUPSeparate(size, a)
    x = LUPSolve(size, l, u, b, p)
    print("L")
    additional.PrintMatrix(l)
    print("U")
    additional.PrintMatrix(u)
    print("L * U")
    additional.PrintMatrix(numpy.array(l) @ numpy.array(u)) # исправить
    print("Ответ")
    additional.PrintVector(x)
    print("Ответ, почлученный при помощи numpy")
    additional.PrintVector(numpy.linalg.solve(a, b))

def LUP(a, b):
    l, u, p = LUPSeparate(len(b), a)
    return LUPSolve(len(b), l, u, b, p)


if __name__ == '__main__':
    size, a, b, _ = additional.GetStatement("1.txt")
    LUPMethod(size, a, b)

'''
задание 3 -- получить обратную матрицу из задания 1 (lup разложение)

задание 4 -- проверить что A * x = lambda * x
'''
