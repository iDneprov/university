import math
import numpy as np
import matplotlib.pyplot as plt

def Function(x, y, y10):
    return 2 * (1 + (math.tan(x) ** 2)) * y

def Answer(x):
    return - math.tan(x)

a = 0
b = math.pi / 6
alpha = 1
beta = 0
delta = 1
gamma = 0
y0 = Answer(0)
y1 = Answer(b)
h = math.pi / 30
error = 0.00001

def p(x):
    return 0

def q(x):
    return - 2 * (1 + (math.tan(x) ** 2))

def F(x):
    return 0


# Алгоритмы
def RungeRombergError(answer):
    k = 0.5
    y1List = [yi for xi, yi in zip(answer[0][0], answer[0][1]) if xi in answer[1][0]]
    y2List = [yi for xi, yi in zip(answer[1][0], answer[1][1]) if xi in answer[0][0]]
    errors = [y1 + (y2 - y1) / (k ** 2 - 1) for y1, y2 in zip(y1List, y2List)]
    xTest = [xi for xi in answer[0][0] if xi in answer[1][0]]
    yTest = [Answer(i) for i in xTest]
    error = 0
    for i in range(len(errors)):
        errors[i] = abs(errors[i] - yTest[i])
        error += errors[i]
    return math.sqrt(error)

def g(x, y, k):
    return k

def RungeKutta(x, y0, y_der, h, f):
    y = [y0]
    k = [y_der]
    for i in range(len(x) - 1):
        K1 = h * g(x[i], y[i], k[i])
        L1 = h * f(x[i], y[i], k[i])
        K2 = h * g(x[i] + 0.5 * h, y[i] + 0.5 * K1, k[i] + 0.5 * L1)
        L2 = h * f(x[i] + 0.5 * h, y[i] + 0.5 * K1, k[i] + 0.5 * L1)
        K3 = h * g(x[i] + 0.5 * h, y[i] + 0.5 * K2, k[i] + 0.5 * L2)
        L3 = h * f(x[i] + 0.5 * h, y[i] + 0.5 * K2, k[i] + 0.5 * L2)
        K4 = h * g(x[i] + h, y[i] + K3, k[i] + L3)
        L4 = h * f(x[i] + h, y[i] + K3, k[i] + L3)
        y.append(y[i] + (K1 + 2 * K2 + 2 * K3 + K4) / 6)
        k.append(k[i] + (L1 + 2 * L2 + 2 * L3 + L4) / 6)
    return x, y, k

def Shooting(a, b, y0, y1, h, error):
    nPrev = 1
    n = 0.8
    y10 = nPrev
    x = [i for i in np.arange(a, b + h, h)]
    answerPrev = RungeKutta(x, nPrev, y10, h, Function)
    y10 = n
    answer = RungeKutta(x, n, y10, h, Function)

    while abs(answer[1][-1] - y1) > error: # условие остановки
        x, y = answerPrev[0], answerPrev[1]
        phiPrev = y[-1] - y1
        x, y = answer[0], answer[1]
        phi = y[-1] - y1
        n, nPrev = n - (n - nPrev) / (phi - phiPrev) * phi, n # метод секущих
        answerPrev = answer
        y10 = n
        answer = RungeKutta(x, y0, y10, h, Function)
    return answer

def RunThrough(a, b, c, d, shape):
    p = [-c[0] / b[0]]
    q = [d[0] / b[0]]
    x = [0] * (shape + 1)
    for i in range(1, shape):
        p.append(-c[i] / (b[i] + a[i] * p[i - 1]))
        q.append((d[i] - a[i] * q[i - 1]) / (b[i] + a[i] * p[i - 1]))
    for i in reversed(range(shape)):
        x[i] = p[i] * x[i + 1] + q[i]
    return x[:-1]

def FiniteDifference(a, b, alpha, beta, delta, gamma, y0, y1, h):
    n = int((b - a) / h)
    x = [i for i in np.arange(a, b + h, h)]
    aDiagonal = [0] + [1 - p(x[i]) * h / 2 for i in range(0, n - 1)] + [-gamma]
    bDiagonal = [alpha * h - beta] + [q(x[i]) * h ** 2 - 2 for i in range(0, n - 1)] + [delta * h + gamma]
    cDiagonal = [beta] + [1 + p(x[i]) * h / 2 for i in range(0, n - 1)] + [0]
    d = [y0 * h] + [F(x[i]) * h ** 2 for i in range(0, n - 1)] + [y1 * h]
    y = RunThrough(aDiagonal, bDiagonal, cDiagonal, d, len(aDiagonal))
    return x, y

# Функции вывода
def ShootingMethod(a, b, y0, y1, h, error):
    print('Метод стрельбы')
    answer = Shooting(a, b, y0, y1, h, error)
    answerH2 = Shooting(a, b, y0, y1, h / 2, error)
    print('Точность результата согласно методу Рунге-Ромберга =', RungeRombergError((answer, answerH2)))
    print()
    return answer, answerH2

def FiniteDifferenceMethod(a, b, alpha, beta, delta, gamma, y0, y1, h):
    print('Конечно-разностный метод')
    answer = FiniteDifference(a, b, alpha, beta, delta, gamma, y0, y1, h)
    answerH2 = FiniteDifference(a, b, alpha, beta, delta, gamma, y0, y1, h)
    print('Точность результата согласно методу Рунге-Ромберга =', RungeRombergError((answer, answerH2)))
    print()
    return answer, answerH2

def DrawChart(answer1, answer2):
    xp = [i for i in np.arange(a, b + h, h)]
    yp = list(map(Answer, xp))

    fig = plt.figure()
    fg = fig.add_subplot()

    fg.plot(xp, yp, 'o')

    line1, = fg.plot(answer1[0][0], answer1[0][1])
    line2, = fg.plot(answer2[0][0], answer2[0][1])
    line3, = fg.plot(xp, yp)

    fg.legend((line1, line2, line3), ("М. Стрельбы", "М. Конечно-разностный", "Точное решение"), loc='best')
    fg.grid()
    plt.show()

if __name__ == '__main__':
    shootingAnswer = ShootingMethod(a, b, y0, y1, h, error)
    finiteDifferenceAnswer = FiniteDifferenceMethod(a, b, alpha, beta, delta, gamma, y0, y1, h)
    DrawChart(shootingAnswer, finiteDifferenceAnswer)
