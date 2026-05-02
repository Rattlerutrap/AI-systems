import csv
import random
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def printPlot1(neighbors, accuracies):
    plt.figure(figsize=(8, 6))
    plt.plot(neighbors, accuracies, linewidth=2)
    plt.xlabel('Количество соседей')
    plt.ylabel('Точность')
    plt.title('График зависимости ошибки классификации от количества ближайших соседей')
    plt.grid(alpha=0.3)
    plt.savefig('plot1.png') 

def getAccuracyTroughMetric(i, metric, data, target):
    trainSize = 150
    accuracies = []
    for j in range(i):
        neigh = KNeighborsClassifier(n_neighbors=5, metric=metric)

        x_train = data[:trainSize]
        y_train = target[:trainSize]

        x_test = data[trainSize:]
        y_test = target[trainSize:]

        neigh.fit(x_train, y_train)

        y_pred = neigh.predict(x_test)

        total = len(x_test)
        correct = (y_pred == y_test).sum()
        wrong = total - correct
        accuracy = correct/total

        accuracies.append(accuracy)

        for i in range(len(data)):
            data[i].append(target[i])
        random.shuffle(data)
        target = []
        for i in data:
            target.append(i[-1])
            i.pop()

    return accuracies

def getDifferentNeighbors(numN, data, target):
    trainSize = 150
    for i in range(1, numN):
        neigh = KNeighborsClassifier(n_neighbors=i)

        x_train = data[:trainSize]
        y_train = target[:trainSize]

        x_test = data[trainSize:]
        y_test = target[trainSize:]

        neigh.fit(x_train, y_train)

        y_pred = neigh.predict(x_test)

        total = len(x_test)
        correct = (y_pred == y_test).sum()
        wrong = total - correct
        accuracy = correct/total

        accuracies.append(accuracy)
        neighbors.append(i)

        # print(f'total: {total}, correct: {correct}, wrong: {wrong}, accuracy: {accuracy}, neighboors: {i}')
    return accuracies, neighbors


FILENAME = 'glass.csv'

data = []

with open(FILENAME, "r", newline="") as file:
    reader = csv.reader(file)
    next(reader)
    for row in reader:
        data.append(list(map(float, row[1:])))

random.seed(0)

random.shuffle(data)
target = []
for i in data:
    target.append(i[-1])
    i.pop()
    
scaler = StandardScaler()
data_normalized = scaler.fit_transform(data)
data_normalized = data_normalized.tolist()


accuracies = []
neighbors = []

accuracies, neighbors = getDifferentNeighbors(30, data_normalized, target)

printPlot1(neighbors, accuracies)

cosineMetric = getAccuracyTroughMetric(10, 'cosine', data_normalized, target)
manhattanMetric = getAccuracyTroughMetric(10, 'manhattan', data_normalized, target)
euclideanMetric = getAccuracyTroughMetric(10, 'euclidean', data_normalized, target)
ar = list(range(1, 11))

plt.figure(figsize=(8, 6))
plt.plot(ar, cosineMetric, label='Cosine', linewidth=2)
plt.plot(ar, manhattanMetric, label='Manhattan', linewidth=2)
plt.plot(ar, euclideanMetric, label='Euclidean', linewidth=2)
plt.xlabel('№ эксперимента')
plt.ylabel('Точность')
plt.title('График зависимости точности классификации от метрики расстояния')
plt.grid(alpha=0.3)
plt.savefig('plot2.png') 

trainSize = 150

neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(data_normalized, target)
y_pred = neigh.predict(scaler.transform([[1.516, 11.7, 1.01, 1.19, 72.59, 0.43, 11.44, 0.02, 0.1]]))
print(y_pred)
    
