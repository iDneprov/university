import math
import numpy as np

from collections import Counter

class LogisticRegression():
    @staticmethod
    def sigmoid(x):
        return 1.0 / (1 + np.exp(-x))

    def fit(self, x, y):
        x = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)
        self.theta = np.zeros(x.shape[1])
        # находим оптимальный разделитель
        for i in range(1000):
            h = self.sigmoid(np.dot(x, self.theta))
            self.theta -= 0.01 * (np.dot(x.T, (h - y)) / y.size)

    def predict(self, x):
        # находим вероятность принадлежности конкретному классу (расстояние до разделителя биективно приобразованное в значение в отрезке от 0 до 1) 0.5 -- точка находится на разделителе
        return self.sigmoid(np.dot(np.concatenate((np.ones((x.shape[0], 1)), x), axis=1), self.theta)) >= 0.5

class KNeighborsClassifier():
    def __init__(self, k=5):
        if k % 2 == 0:
            raise IOError("K should be odd number!")
        self.k = k

    def fit(self, x, y):
        self.x = x
        self.y = y.reshape((y.shape[0], 1))

    def distance(self, x):
        return np.sqrt(((self.x - x) ** 2).sum(1))

    def predict(self, x):
        yPredicted = np.zeros((x.shape[0], 1))
        # сортируем точки по координатам и берем К ближайших, если больше половины точек принадлежат классу 1, то прописываем это в массив предсказанных значений
        for i in range(x.shape[0]):
            yDistanceSorted = self.y[np.argsort(self.distance(x[i]))].flatten()
            if np.add.reduce(yDistanceSorted[:self.k]) > self.k / 2:
                yPredicted[i] = 1
        return yPredicted

class Node():
    def __init__(self, classPrediction):
        self.classPrediction = classPrediction
        self.left = None
        self.right = None
        self.iFeature = 0
        self.threshold = 0


class DecisionTreeClassifier():
    def __init__(self, hightMax=1, forRandomForest=False):
        if hightMax < 1 or hightMax != int(hightMax):
            raise IOError("Max Hight should be positive integer number!")
        self.hightMax = hightMax
        self.forRandomForest = forRandomForest

    def fit(self, x, y, featuresMax=None):
        self.classesNumber = len(set(y))
        if self.forRandomForest:
            # по сути делаем шафл x и y (какие-то признаки могут повторяться, а какие-то отсутствовать)
            i = np.random.choice(x.shape[0], x.shape[0])
            x = x[tuple([i])]
            y = y[tuple([i])]
            # получаем число признаков, которые будем использовать
            featuresNumber = np.sqrt(x.shape[1]).astype(int) if featuresMax is None else featuresMax
        else:
            featuresNumber = x.shape[1]
        self.features = np.sort(np.random.choice(x.shape[1], featuresNumber, replace=False))
        self.root = self.makeTree(x, y)

    def findSplitThreshold(self, x, y):
        learningSetSize = y.size
        if learningSetSize <= 1:
            return None, None
        # вычисляем начальный коэффициент Джини
        parentsNumber = [np.sum(y == i) for i in range(self.classesNumber)]
        giniBest = 1.0 - sum((n / learningSetSize) ** 2 for n in parentsNumber)
        indexBest = None
        thresholdBest = None
        for indexCurrent in self.features:
            thresholds, classes = zip(*sorted(zip(x[:, indexCurrent], y)))
            leftNumbers = [0] * self.classesNumber
            rightNumbers = parentsNumber.copy()
            # подбираем класс для разбиения
            for currentClass, i in zip(classes, range(1, learningSetSize)):
                leftNumbers[currentClass] += 1
                rightNumbers[currentClass] -= 1
                # вычисляем новый коэффициент Джини и изменяем параметры, которые относятся к лучшему разбиению
                giniCurrent = (i * (1.0 - sum((leftNumbers[x] / i) ** 2 for x in range(self.classesNumber))) + (learningSetSize - i) * (1.0 - sum((rightNumbers[x] / (learningSetSize - i)) ** 2 for x in range(self.classesNumber)))) / learningSetSize
                if thresholds[i] != thresholds[i - 1] and giniCurrent < giniBest:
                    giniBest = giniCurrent
                    indexBest = indexCurrent
                    thresholdBest = (thresholds[i] + thresholds[i - 1]) / 2
        return indexBest, thresholdBest

    def makeTree(self, x, y, hightCurrent=0):
        # создаем корень
        classPrediction = np.argmax([np.sum(y == i)for i in range(self.classesNumber)])
        node = Node(classPrediction=classPrediction)
        # наращиваем высоту дерева, пока не достигнем максимума
        if hightCurrent < self.hightMax:
            indexCurrent, threshold = self.findSplitThreshold(x, y)
            # или не кончатся параметры
            if indexCurrent is not None:
                # делим значения на те, которые будут относиться к правому и левому поддереву
                isRight = x[:, indexCurrent] > threshold
                xRight = x[isRight]
                yRight = y[isRight]
                xLeft = x[~isRight]
                yLeft = y[~isRight]
                node.iFeature = indexCurrent
                node.threshold = threshold
                # строим эти поддеревья
                node.right = self.makeTree(xRight, yRight, hightCurrent + 1)
                node.left = self.makeTree(xLeft, yLeft, hightCurrent + 1)
        return node

    def predict(self, x):
        result = []
        # обрабатываем каджый элемент отдельно
        for row in x:
            node = self.root
            # поднимаемся по дереву, пока не упремся в лист
            while node.right:
                if row[node.iFeature] > node.threshold:
                    node = node.right
                else:
                    node = node.left
            # записываем класс листа
            result.append(node.classPrediction)
        return result

class RandomForestClassifier():
    def __init__(self, hightMax=5, treesNumber=200, featuresMax=None):
        self.hightMax = hightMax
        self.featuresMax = featuresMax
        self.treesNumber = treesNumber
        self.trees = []

    def fit(self, x, y):
        # создаем treesNumber деревьев и обучаем их
        for i in range(self.treesNumber):
            self.trees.append(DecisionTreeClassifier(self.hightMax, forRandomForest=True))
            self.trees[i].fit(x, y)

    def predict(self, x):
        predictions = np.zeros((self.treesNumber, x.shape[0]))
        result = np.zeros(x.shape[0])
        # получаем классификацию от всех деревьев
        for i in range(self.treesNumber):
            predictions[i] = self.trees[i].predict(x)
        # и выбираем наиболее часто встречающийся класс
        for i in range(len(result)):
            result[i] = Counter(predictions[:, i]).most_common(1)[0][0]
        return result
