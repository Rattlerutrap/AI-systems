import numpy as np
import random



n_neg = 20
mean_neg = [10, 14]
std_neg = 3

n_pos = 80
mean_pos = [19, 16]
std_pos = 4

np.random.seed(42)
X_neg = np.random.normal(mean_neg, std_neg, size=(n_neg, 2))
X_pos = np.random.normal(mean_pos, std_pos, size=(n_pos, 2))


dataSet = []

for i in range(len(X_neg)):
    dataSet.append([X_neg[i][0], X_neg[1], -1])

for i in range(len(X_pos)):
    dataSet.append([X_pos[i][0], X_pos[1], 1])

random.shuffle(dataSet)
target = []
for i in dataSet:
    target.append(i[-1])
    i.pop()

print(target)