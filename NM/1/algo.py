import additional
import math
import copy
import numpy as np
from numpy.linalg import norm, solve, inv


def Swap(matrix, i, j):
    matrix[i], matrix[j] = matrix[j].copy(), matrix[i].copy()

def create_pMatrixatrix(matrix, p):
    pMatrix = matrix.copy()

    for i, j in p:
        pMatrix[i], pMatrix[j] = pMatrix[j], pMatrix[i]
    return pMatrix


def LUPSeparate(size, a):
    l = [[0.0] * size for i in range(size)]

    p = []
    u = copy.deepcopy(a)
    for j in range(size - 1):
        tmp = [(u[i][j], i) for i in range(j, size)]
        idx = max(tmp, key=lambda x: abs(x[0]))[1]
        if idx != j:
            p.append((j, idx))
            Swap(u, j, idx)

    for i in range(size):
        l[i][i] = 1
        for j in range(i + 1, size):
            l[j][i] = u[j][i] / u[i][i]
            for k in range(i + 1, size):
                u[j][k] -= l[j][i] * u[i][k]

    for i in range(1, size):
        for j in range(i):
            u[i][j] = 0.0

    return l, u, p


def LUPSolve(size, l, u, b, p):
    z = [0 for i in range(size)]
    x = [0 for i in range(size)]

    pMatrix = b.copy()

    for i, j in p:
        pMatrix[i], pMatrix[j] = pMatrix[j], pMatrix[i]

    for i in range(size):
        z[i] = pMatrix[i] - sum(l[i][j] * z[j] for j in range(i))

    for i in reversed(range(size)):
        n = sum(u[i][j] * x[j] for j in range(i + 1, size))
        x[i] = (z[i] - n) / u[i][i]
    return x


def tridiagonal(size, a, b):
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

def transformToEqual(size, a, b):
    alpha = [[0.0] * size for i in range(size)]
    beta = [0.0 for i in range(size)]
    aCoppy = copy.deepcopy(a)

    swaps = []
    for i in range(size):
        if a[i][i] == 0:
            for j in range(size):
                if a[j][i] != 0 and a[i][j] != 0:
                    swaps.append(i, j)
                    additional.Swap()


    for i in range(size):
        beta[i] = b[i] / a[i][i]
        for j in range(size):
            alpha[i][j] = -a[i][j] / a[i][i]
        alpha[i][i] = 0
    return alpha, beta


def Iteration(size, a, b, precision=0.01):
    a = [row.copy() for row in a]
    alpha, beta = transformToEqual(size, a, b)

    q = additional.matrixNorm(alpha)
    cur = beta.copy()
    last = []
    while True:
        last = cur
        mult = additional.mv_mult(size, alpha, cur)
        cur = additional.vv_add(size, beta, mult)
        norma = additional.vectorNorm(
            size, additional.SubtractVectors(size, cur, last))
        if norma * q / (1 - q) <= precision:
            break

    return cur


def get_norm(a, alpha, S, C):
    return norm(C, np.inf) / (1. - norm(S, np.inf))


def zeidel_method(size, a, b, precision=0.01):
    alpha, beta = transformToEqual(a, b)

    alpha = np.array(alpha)
    beta = np.array(beta)

    b = np.tril(alpha, -1)

    K = alpha - b

    T1 = inv(
        np.eye(size, size) - b) @ K

    T2 = inv(
        np.eye(size, size) - b) @ beta

    x = T2
    C = get_norm(a, alpha, T1, K)

    cov = True
    while cov:
        X_next = T2 + T1 @  x
        if C * norm(X_next - x, np.inf) <= precision:
            cov = False
        x = X_next

    return X_next


def rotate_jacobi(size, a, precision=0.01):
    Ak = [row.copy() for row in a]

    idx = range(size)
    u = [[0. if i != j else 1. for i in idx] for j in idx]

    cov = False
    while not cov:
        ik, jk = 0, 1
        for i in range(size - 1):
            for j in range(i + 1, size):
                if abs(Ak[i][j]) > abs(Ak[ik][jk]):
                    ik, jk = i, j

        if Ak[ik][ik] == Ak[jk][jk]:
            phi = math.pi / 4
        else:
            phi = 0.5 * math.atan(
                2 * Ak[ik][jk] / (Ak[ik][ik] - Ak[jk][jk]))

        Uk = [[0. if i != j else 1. for i in idx] for j in idx]
        Uk[ik][jk] = math.sin(phi)
        Uk[jk][ik] = -Uk[ik][jk]

        Uk[ik][ik] = math.cos(phi)
        Uk[jk][jk] = math.cos(phi)

        tmp = additional.mm_mult(Uk, Ak)

        Uk[ik][jk], Uk[jk][ik] = Uk[jk][ik], Uk[ik][jk]

        Ak = additional.mm_mult(tmp, Uk)
        u = additional.mm_mult(u, Uk)

        accum = 0
        for i in range(size - 1):
            for j in range(i + 1, size):
                accum += Ak[i][j] ** 2

        avg = math.sqrt(accum)
        if avg < precision:
            cov = True

    return [Ak[i][i] for i in range(size)], u


def sign(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0


def QR_decomposition(size, a, precision=0.01):
    Ak = [row.copy() for row in a]

    E = additional.e(size)
    Q = additional.e(size)

    for i in range(size):
        V = [0 for _ in range(size)]

        V[i] = Ak[i][i] + sign(Ak[i][i]) * math.sqrt(
            sum(Ak[j][i] ** 2 for j in range(i, size)))

        for k in range(i + 1, size):
            V[k] = Ak[k][i]

        Vt = [V]
        V = [[V[i]] for i in range(size)]

        M = additional.mm_mult(V, Vt)
        C = additional.mm_mult(Vt, V)[0][0]
        for j in range(size):
            for k in range(size):
                M[j][k] /= C
                M[j][k] *= 2

        Hk = additional.SubstractMaxrix(E, M)
        Q = additional.mm_mult(Q, Hk)
        Ak = additional.mm_mult(Hk, Ak)

    return Q, Ak


def roots(a, types):
    n = len(a)
    a = np.array(a)
    soLUPtion = []
    k = 0
    for t in types:
        if t == 'real':
            soLUPtion.append(a[k, k])
        else:

            A11 = a[k, k]
            A12 = A21 = A22 = 0

            if k + 1 < n:
                A12 = a[k, k + 1]
                A21 = a[k + 1, k]
                A22 = a[k + 1, k + 1]

            soLUPtion.extend(np.roots(
                (1, -A11 - A22, A11 * A22 - A12 * A21)))
            k += 1
        k += 1
    return soLUPtion





def check(matrix, precision=0.01):
    n = len(matrix)
    check = []
    k = 0

    def square_norm(x):
        return sum(e ** 2 for e in x) ** 0.5

    def get_coLUPmn(a, k):
        return [a[i][k] for i in range(k+1, len(a))]

    while k < n:
        if square_norm(get_coLUPmn(matrix, k)) <= precision:
            check.append('real')
        elif square_norm(get_coLUPmn(matrix, k + 1)) <= precision:
            check.append('img')
            k += 1
        else:
            check.append(None)
        k += 1
    return check


def QR_method(size, a, precision=0.01):
    Ak = [row.copy() for row in a]
    it, max_it = 0, 100

    step = True

    while it < max_it:
        Q, R = QR_decomposition(size, Ak)
        Ak = additional.mm_mult(R, Q)
        additional.matrix_print(Q, header=f"a{it}")

        types = check(Ak, precision)
        if all(types):
            if step:
                step = False
            else:
                return roots(Ak, types)
        it += 1

    return None
