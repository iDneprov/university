statement = {"x*": 1.0, "x": [0.0, 0.5, 1.0, 1.5, 2.0], "y": [0.0, 0.97943, 1.8415, 2.4975, 2.9093]}

def GetDerivativesValues(x, y, dot):
    i = 0
    while x[i + 1] < dot:
        i += 1

    while i + 1 > len(x) + 1:
        i -= 1

    firstDerivative = (y[i+1] - y[i])/(x[i+1] - x[i]) + ((y[i+2] - y[i+1])/(x[i+2] - x[i+1]) - (y[i+1] - y[i])/(x[i+1] - x[i])) * (2 * dot - x[i] - x[i+1]) / (x[i+2] - x[i])
    secondDerivative = 2 * ((y[i+2] - y[i+1])/(x[i+2] - x[i+1]) - (y[i+1] - y[i])/(x[i+1] - x[i])) / (x[i+2] - x[i])
    print('Численное диффиринцирование функции')
    print()
    print(f'F\'({dot}) = {firstDerivative}.')
    print(f'F\'\'({dot}) = {secondDerivative}.')
    return firstDerivative, secondDerivative

if __name__ == '__main__':
    GetDerivativesValues(statement['x'], statement['y'], statement['x*'])