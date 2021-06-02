import numpy
import math
import additional
import copy

def Rotation(size, aOriginal, precision):
    a = copy.deepcopy(aOriginal)
    # единичная матрица
    e = [[0 if i != j else 1 for i in range(size)] for j in range(size)]
    uMultiply = copy.deepcopy(e)
    n = 0
    while True:
        # находим индекс максимального по модулю элемента в матрице
        iMax = 0
        jMax = 1
        for i in range(size - 1):
            for j in range(i + 1, size):
                if abs(a[i][j]) > abs(a[iMax][jMax]):
                    iMax = i
                    jMax = j
        # вычисляем угол поворота
        if a[iMax][iMax] == a[jMax][jMax]:
            angle = math.pi / 4
        else:
            angle = 0.5 * math.atan(2 * a[iMax][jMax] / (a[iMax][iMax] - a[jMax][jMax]))
        # создаем матриуц вращения
        u = copy.deepcopy(e)
        u[iMax][iMax] = math.cos(angle)
        u[iMax][jMax] = -math.sin(angle)
        u[jMax][iMax] = math.sin(angle)
        u[jMax][jMax] = math.cos(angle)
        # получаем новую версию матрицы
        tmp = additional.MultiplyMatrix(a, u)
        u[iMax][jMax], u[jMax][iMax] = u[jMax][iMax], u[iMax][jMax] # транспонируем матрицу вращения
        a = additional.MultiplyMatrix(u, tmp)
        uMultiply = additional.MultiplyMatrix(u, uMultiply)
        # проверяем условие останова
        accum = 0
        personalValues = [a[i][i] for i in range(size)]
        print("Собственные значения на %i-ой итерации" % n)
        n += 1
        additional.PrintVector(personalValues)
        for i in range(size - 1):
            for j in range(i + 1, size):
                accum += a[i][j] ** 2 # получаем половинную сумму квадратов недиагональных элементов
        if math.sqrt(accum) < precision:
            break
    additional.Transpose(uMultiply)
    return personalValues, uMultiply

def RotationMethod(size, a, precision):
    ac = copy.deepcopy(a)
    print("Лабораторная работа 1.4")
    print("Метод вращений")
    print()
    print("Матрица")
    additional.PrintMatrix(a)
    x, lambd = Rotation(size, a, precision)
    print("Ответ")
    print()
    print("Собственные значения")
    additional.PrintVector(x)
    print("Собственные векторы")
    additional.PrintMatrix(lambd)
    print("Проверка")
    a = numpy.array(a)
    lambd = numpy.array(lambd)
    additional.PrintMatrix(numpy.dot(a, lambd))
    print('=\n')
    additional.PrintMatrix(lambd * x)
    '''
    for i in range(size):
        for j in range(size):
            an = numpy.matrix(a)
            xn = numpy.array(x[j])
            ln = lambd[i]
            print((an - [[ln if i == j else 0 for j in range(size)] for i in range(size)]) @ xn)
    '''
    print("Ответ, почлученный через numpy.linalg")
    x, lambd = numpy.linalg.eig(ac)
    print()
    print("Собственные значения")
    additional.PrintVector(x)
    print("Собственные векторы")
    additional.PrintMatrix(lambd)

if __name__ == '__main__':
    size, a, b, precision = additional.GetStatement("4.txt")
    RotationMethod(size, a, precision)
