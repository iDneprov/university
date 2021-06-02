import matplotlib.pyplot as plt
import math
import numpy as np

def Answer(x):
    return np.cos(np.sin(x)) + np.sin(np.cos(x))

def AnswerDerevative(x):
        return np.sin(x) * (-np.cos(np.cos(x))) - np.sin(np.sin(x)) * np.cos(x)

def Function(x, y, y1):
    return -y1 * np.tan(x) - y * (np.cos(x) ** 2)

def AbsoluteError(x, y):
    error = 0.0
    for i in range(len(x)):
        error += (y[i] - Answer(x[i])) ** 2
    return error ** 0.5

def RungeRombergError(y1, y2, p):
    error = 0.0
    for i in range(len(y2)):
        error += (y1[i*2] - y2[i]) ** 2
    return (error ** 0.5) / (2 ** p - 1)

def GetXs(a, b, h):
    return list(np.arange(a, b+h, h))

a = 0
b = 1
h = 0.1
y0 = Answer(0)
y10 = AnswerDerevative(0)

# Алгоритмы
def Eiler(x, y0, y10, h, f = Function):
    y = [y0]
    y1 = [y10]
    for i in range(len(x) - 1):
        y.append(y[i] + h*y1[i])
        y1.append(y1[i] + h*f(x[i], y[i], y1[i]))
    return y, y1

def EulerKodhiItter(x, y0, y10, h, f = Function):
    y = [y0]
    y1 = [y10]

    for k in range(len(x) - 1):
        yk = y[k] + h*y1[k]
        zk = y1[k] + h*f(x[k], y[k], y1[k])
        for i in range(4):
            ykk = yk
            zkk = zk
            yk = y[k] + h*(y1[k] + zkk) / 2
            zk = y1[k] + h*(f(x[k], y[k], y1[k]) + f(x[k+1], ykk, zkk)) / 2
        y.append(yk)
        y1.append(zk)
    return y, y1

def RungeKutta(x, y0, y10, h, f = Function):
    y = [y0]
    y1 = [y10]
    for i in range(len(x) - 1):
        K1 = h * y1[i]
        L1 = h * f(x[i], y[i], y1[i])
        K2 = h * (y1[i] + L1 / 2)
        L2 = h * f(x[i] + h/2, y[i] + K1/2, y1[i] + L1/2)
        K3 = h * (y1[i] + L2 / 2)
        L3 = h * f(x[i] + h/2, y[i] + K2/2, y1[i] + L2/2)
        K4 = h * (y1[i] + L3)
        L4 = h * f(x[i] + h, y[i] + K3, y1[i] + L3)
        y.append(y[i] + (K1 + 2*K2 + 2*K3 + K4)/6)
        y1.append(y1[i] + (L1 + 2*L2 + 2*L3 + L4)/6)
    return y, y1

def Adams(x, y0, y10, h, f = Function):
    y, y1 = RungeKutta(x[:4], y0, y10, h, f)
    for i in range(3, len(x)-1):
        y.append(y[i] + h*(55*y1[i] - 59*y1[i-1] + 37*y1[i-2] - 9*y1[i-3]) / 24)
        y1.append(y1[i] + h*(55*f(x[i], y[i], y1[i]) - 59*f(x[i-1], y[i-1], y1[i-1]) + 37*f(x[i-2], y[i-2], y1[i-2]) - 9*f(x[i-3], y[i-3], y1[i-3])) / 24)
    return y, y1


# Функции вывода
def EulerMethod():
    x = GetXs(a, b, h)
    xh2 = GetXs(a, b, h / 2)
    print('Явный метод Эйлера')
    y, y1 = Eiler(x, y0, y10, h)
    print('Абсолютная погрешность:', AbsoluteError(x, y))
    p = 1
    yr, zr = Eiler(xh2, y0, y10, h / 2)
    print('Погрешность согласно методу Рунге-Ромберга:', RungeRombergError(yr, y, p))
    print()

def EulerKodhiItterMethod():
    x = GetXs(a, b, h)
    xh2 = GetXs(a, b, h / 2)
    print('Явный итерационнный метод Эйлера-Коши')
    y, y1 = EulerKodhiItter(x, y0, y10, h)
    print('Абсолютная погрешность:', AbsoluteError(x, y))
    p = 1
    yr, zr = EulerKodhiItter(xh2, y0, y10, h / 2)
    print('Погрешность согласно методу Рунге-Ромберга:', RungeRombergError(yr, y, p))
    print()

def RungeKuttaMethod():
    x = GetXs(a, b, h)
    xh2 = GetXs(a, b, h / 2)
    print('Метод Рунге-Кутты')
    y, y1 = RungeKutta(x, y0, y10, h)
    print('Абсолютная погрешность:', AbsoluteError(x, y))
    p = 4
    yr, zr = RungeKutta(xh2, y0, y10, h / 2)
    print('Погрешность согласно методу Рунге-Ромберга:', RungeRombergError(yr, y, p))
    print()

def AdamsMethod():
    x = GetXs(a, b, h)
    xh2 = GetXs(a, b, h / 2)
    print('Метод Адамса')
    y, y1 = Adams(x, y0, y10, h)
    print('Абсолютная погрешность:', AbsoluteError(x, y))
    p = 4
    yr, zr = Adams(xh2, y0, y10, h / 2)
    print('Погрешность согласно методу Рунге-Ромберга:', RungeRombergError(yr, y, p))
    print()

def DrawChart():
    xp = GetXs(a, b, h)
    xi = GetXs(a, b, 0.0005)

    yp1, y1p1 = Eiler(xp, y0, y10, h)
    yp2, y1p2 = RungeKutta(xp, y0, y10, h)
    yp3, y1p3 = Adams(xp, y0, y10, h)
    yp5, y1p5 = EulerKodhiItter(xp, y0, y10, h)

    y4 = list(map(Answer, xi))
    yp = list(map(Answer, xp))

    fig = plt.figure()
    fg = fig.add_subplot()

    fg.plot(xp, yp, 'o')

    line1, = fg.plot(xp, yp1)
    line2, = fg.plot(xp, yp2)
    line3, = fg.plot(xp, yp3)
    line4, = fg.plot(xi, y4)
    line5, = fg.plot(xp, yp5)

    fg.legend((line1, line5, line2, line3, line4), ("Эйлер", "Иттер Эейлер-Коши", "Рунге-Кутта", "Адамс", "Точное решение"))
    fg.grid()
    plt.show()

if __name__ == "__main__":
    print("В задании y(0) и y'(0) указаны не верно.")
    print("Я никак не мог понять, что за фигня творится с точностью.")
    print("И когда я уже отчаялся что-то починить, я пошёл строить график и всё понял.")
    print("Я пересчитал их значения:")
    print("y(0) =", y0)
    print("y'(0) =", y10)
    print()
    EulerMethod()
    EulerKodhiItterMethod()
    RungeKuttaMethod()
    AdamsMethod()
    DrawChart()
