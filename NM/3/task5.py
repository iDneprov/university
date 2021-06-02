def f(x):
    return x ** 2 / (x ** 2 + 16)

statement = {"x0": 0, "xk": 2, "h1": 0.5, "h2": 0.25, "val": 0, 'f': f}

def GetX(x0, xk, step=0.1):
    x = x0
    r = []
    while x <= xk:
        r.append(x)
        x += step
    return r

def RectanglesMethod(h, x, f):
    return sum(f((x[i-1] + x[i]) / 2) for i in range(1, len(x))) * h

def TrapeziumsMethod(h, x, f):
    return (f(x[0]) / 2 + sum(f(x[i]) for i in range(1, len(x) - 1)) + f(x[len(x) - 1])) * h

def SimpsonsMethod(h, x, f):
    return (1 / 3) * (f(x[0]) + sum(4 * f(x[i]) for i in range(1, len(x)-1, 2)) + sum(2 * f(x[i]) for i in range(2, len(x)-1, 2)) + f(x[len(x) - 1])) * h

def RungeRombergPrecision(h1, h2, y1, y2, pow=2):
    return abs((y1 - y2) / ((h2 / h1) ** pow - 1.0))

if __name__ == '__main__':
    print('Численное дифференцирование')
    print()
    methods = (("Метод прямоугольников", RectanglesMethod), ("Метод трапеций", TrapeziumsMethod), ("Метод Симпсона", SimpsonsMethod))
    for (methodName, Method) in methods:
        print(methodName)
        first = Method(statement['h1'], GetX(statement['x0'], statement['xk'], statement['h1']), statement['f'])
        second = Method(statement['h2'], GetX(statement['x0'], statement['xk'], statement['h2']), statement['f'])
        print(f"Ответ с шагом {statement['h1']}: F(x) = {first}")
        print(f"Ответ с шагом {statement['h2']}: F(x) = {second}")
        precision = RungeRombergPrecision(statement['h1'], statement['h2'], first, second, pow=2)
        print("Погрешность согласно методу Рунге-Ромберга равна", precision)
        print()
