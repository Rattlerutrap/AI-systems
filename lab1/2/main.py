import numpy as np
import random
import math
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve, roc_auc_score

def plotPrint(X_neg, X_pos):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(X_neg[:, 0], X_neg[:, 1], color='red', alpha=0.7)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Класс -1 (20 точек)')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.scatter(X_pos[:, 0], X_pos[:, 1], color='blue', alpha=0.7)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Класс 1 (80 точек)')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

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
    dataSet.append([X_neg[i][0], X_neg[i][1], -1])

for i in range(len(X_pos)):
    dataSet.append([X_pos[i][0], X_pos[i][1], 1])

# plotPrint(X_neg, X_pos)

random.shuffle(dataSet)
target = []
for i in dataSet:
    target.append(i[-1])
    i.pop()


gnb = GaussianNB()

percent = 60

x_train = dataSet[:percent]
y_train = target[:percent]

x_test = dataSet[percent:]
y_test = target[percent:]

gnb.fit(x_train, y_train)

y_pred = gnb.predict(x_test)

y_pred_proba = gnb.predict_proba(x_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba, pos_label=1)
auc = roc_auc_score(y_test, y_pred_proba)

print(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC (AUC = {auc:.3f})', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', label='Случайный')
plt.xlabel('FPR (False Positive Rate)')
plt.ylabel('TPR (True Positive Rate)')
plt.title('ROC-кривая')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

print(f"total: {len(x_test)}, correct: {(y_pred == y_test).sum()}, accuracy: {(y_pred == y_test).sum()/len(x_test)}")

