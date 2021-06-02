import numpy
import additional

def RunThrough(size, a, b):
    m1 = [a[i + 1][i] for i in range(size - 1)]
    m2 = [a[i][i] for i in range(size)]
    m3 = [a[i][i + 1] for i in range(size - 1)]
    m4 = b.copy()

    for i in range(1, size):
        m = m1[i - 1] / m2[i - 1]
        m2[i] = m2[i] - m * m3[i - 1]
        m4[i] = m4[i] - m * m4[i - 1]

    x = m2.copy()
    x[size - 1] = m4[size - 1] / m2[size - 1]

    for i in reversed(range(0, size - 1)):
        x[i] = (m4[i] - m3[i] * x[i + 1]) / m2[i]

    return x

def RunThroughMethod(size, a, b):
    print("Лабораторная работа 1.2")
    print("Метод прогонки")
    print()
    print("Матрица")
    additional.PrintMatrix(a)
    additional.PrintVector(b)
    x = RunThrough(size, a, b)
    print("Ответ")
    additional.PrintVector(x)
    print("Ответ, почлученный при помощи numpy")
    additional.PrintVector(numpy.linalg.solve(a, b))

if __name__ == '__main__':
    size, a, b, _ = additional.GetStatement("2.txt")
    RunThroughMethod(size, a, b)
