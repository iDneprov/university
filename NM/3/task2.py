import matplotlib.pyplot as plt
import numpy as np
import subprocess, sys

statement = {"x*": -0.5, "n": 4, "x": [-3.0, -1.0, 1.0, 3.0, 5.0], "f": [-1.249, -0.7854, 0.7854, 1.2490, 1.3734]}

def PrintCubicSplineTable(spline, x):
    title = "i. | [x_i-1, x_i] |   a_i   |   b_i   |   c_i   |   d_i   |"
    print(title)
    print('-' * len(title))
    for i in range(1, len(x)):
        print("{}. |   [{:2.0f},{:2.0f}]    | {:7.4f} | {:7.4f} | {:7.4f} | {:7.4f} |".format(i, x[i-1], x[i], spline[i-1][1], spline[i-1][2], spline[i-1][3], spline[i-1][4]))
    print()

def GetC(a, b, c, d): # решение системы
    a = a.copy()
    b = b.copy()
    c = c.copy()
    d = d.copy()
    for i in range(1, len(d)):
        m = a[i - 1] / b[i - 1]
        b[i] = b[i] - m * c[i - 1]
        d[i] = d[i] - m * d[i - 1]

    x = b.copy()
    x[len(d) -1] = d[len(d) - 1] / b[len(d) - 1]

    for i in reversed(range(0, len(d) - 1)):
        x[i] = (d[i] - c[i] * x[i + 1]) / b[i]
    return x

def GetCubicSpline(x, f):
    n = len(x)
    # получаем коэффициенты системы уравнений
    a = [x[i - 1] - x[i - 2] for i in range(3, n)]
    b = [2 * (x[i] - x[i - 2]) for i in range(2, n)]
    c = [(x[i] - x[i - 1]) for i in range(2, n-1)]
    d = [3 * ((f[i] - f[i-1]) / (x[i] - x[i - 1]) - ((f[i-1] - f[i-2]) / (x[i - 1] - x[i - 2]))) for i in range(2, n)]
    # решаем систему (заполняем таблицу коэффициентов)
    aCoeff = [0] + [f[i] for i in range(n-1)]
    bCoeff = [0]
    cCoeff = [0, 0] + GetC(a, b, c, d) # решам систему с С
    dCoeff = [0]
    for i in range(1, n-1):
        bCoeff.append((f[i] - f[i - 1]) / (x[i] - x[i - 1]) - 1/3 * (x[i] - x[i - 1]) * (cCoeff[i + 1] + 2 * cCoeff[i]))
    bCoeff.append((f[n-1] - f[n-2]) / (x[n - 1] - x[n - 2]) - 2/3 * (x[n - 1] - x[n - 2]) * cCoeff[n-1])
    dCoeff += [(cCoeff[i+1] - cCoeff[i]) / (3 * (x[i] - x[i - 1])) for i in range(1, n-1)]
    dCoeff.append(-cCoeff[n-1] / 3 * (x[n - 1] - x[n - 2]))
    # вычисляем коэффициенты сплайна
    spline = []
    for i in range(1, n):
        spline.append([x[i-1], aCoeff[i], bCoeff[i], cCoeff[i], dCoeff[i]])
    return spline

def CalculateSplineValue(spline, x, val):
    fields = [(x[i-1], x[i]) for i in range(1, len(x))]
    k = 0
    for i, f in enumerate(fields):
        if val < f[1] and val >= f[0]:
            k = i
            break
    def Calculate(s, x):
        return s[1]+s[2]*(x-s[0])+s[3]*((x-s[0])**2)+s[4]*((x-s[0])**3)
    return Calculate(spline[k], val)

def DrawChart(spline, fileName, step=0.1):
    x = np.arange(statement['x'][0], statement['x'][-1] + 0.1, step)
    y = [CalculateSplineValue(spline, statement['x'], val) for val in x]
    y[-1] = 1.3734

    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.plot(statement['x'], statement['f'], 'ro')
    ax.plot(statement['x'], statement['f'])

    ax.grid()

    if fileName:
        fig.savefig(fileName)
        plt.close(fig)
        opener = "open" if sys.platform == "darwin" else "xdg-open"
        subprocess.call([opener, fileName])

    plt.show()

def CubicSpline():
    spline = GetCubicSpline(statement["x"], statement["f"])
    PrintCubicSplineTable(spline, statement["x"])
    print("Spline({}) = {:5.3f}".format(statement["x*"], CalculateSplineValue(spline, statement["x"], statement["x*"])))
    DrawChart(spline, "CubicSpline.png")

if __name__ == '__main__':
    CubicSpline()
