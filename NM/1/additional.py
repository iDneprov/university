import math
import numpy
from numpy.linalg import norm, solve, inv

def Sign(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0

def GetStatement(fileName='statement.txt'):
    statement = open(fileName)
    precision = float(statement.readline())
    size = int(statement.readline())
    a = []
    for i in range(1, size + 1):
        l = statement.readline().split()
        b = []
        for j in l:
            b.append(int(j))
        a.append(b)

    b = []
    l = statement.readline().split()
    for i in l:
        b.append(int(i))

    return size, a, b, precision

def Swap(matrix, i, j):
    matrix[i], matrix[j] = matrix[j].copy(), matrix[i].copy()

def PrintMatrix(matrix):
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            print('{:6.3f}'.format(float(matrix[i][j])), end=' ')
        print()
    print()

def PrintVector(vector):
    for i in range(len(vector)):
        print(vector[i], end=' ') # print('{:6.5f}'.format(float(vector[i])), end=' ')
    print()
    print()

def MultiplyMatrivOnVector(size, matrix, vector):
    R = [0.0 for _ in range(size)]
    for i in range(size):
        for j in range(size):
            R[i] += matrix[i][j] * vector[j]
    return R

def MultiplyVectors(matrix1, matrix2):
    C = [0.0 for col in range(len(matrix1))]
    for i in range(len(matrix1)):
        for j in range(len(matrix2[0])):
            for k in range(len(matrix1[0])):
                C[i][j] += matrix1[i][k] * matrix2[k][j]
    return C

def MultiplyMatrix(matrix1, matrix2):
    C = [[0.0] * len(matrix2[0]) for i in range(len(matrix1))]
    for i in range(len(matrix1)):
        for j in range(len(matrix2[0])):
            for k in range(len(matrix1[0])):
                C[i][j] += matrix1[i][k] * matrix2[k][j]
    return C

def SubtractVectors(size, vector1, vector2):
    return [vector1[i] - vector2[i] for i in range(size)]

def SubstractMaxrix(matrix1, matrix2):
    return [[matrix1[i][j] - matrix2[i][j] for i in range(
        len(matrix1[j]))] for j in range(len(matrix1))]

def SummVectors(size, vector1, vector2):
    return [vector1[i] + vector2[i] for i in range(size)]

def VectorNorm(size, vector):
    return sum(abs(vector[i]) for i in range(size))

def NPNorm(a, alpha, S, C):
    return norm(C, numpy.inf) / (1. - norm(S, numpy.inf))

def Transpose(matrix):
    for i in range(len(matrix) - 1):
        for j in range(i + 1, len(matrix)):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]

def MatrixNorm(matrix):
    max = -10**10
    for i in matrix:
        summ = 0
        for j in i:
            summ += j
        if summ > max:
            max = summ
    return max

def GetVectorOfCoeffs(a, k):
    return [a[i][k] for i in range(k+1, len(a))]

def SNorm(x):
    return sum(i ** 2 for i in x) ** 0.5

def HaveNoRoot(rootType):
    if len(rootType) == 0:
        return True
    for type in rootType:
        if type == 'not root':
            return False
    return True
