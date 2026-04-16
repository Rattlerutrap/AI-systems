import csv
import random
from sklearn.neighbors import KNeighborsClassifier


FILENAME = 'glass.csv'

data = []

with open(FILENAME, "r", newline="") as file:
    reader = csv.reader(file)
    next(reader)
    for row in reader:
        data.append(list(map(float, row[1:])))

random.shuffle(data)
target = []
for i in data:
    target.append(i[-1])
    i.pop()
    



trainSize = 150
for i in range(1, 30):
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

    print(f'total: {total}, correct: {correct}, wrong: {wrong}, accuracy: {correct/total}, neighboors: {i}')

