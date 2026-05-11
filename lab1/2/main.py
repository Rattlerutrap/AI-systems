import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, confusion_matrix, classification_report

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

plotPrint(X_neg, X_pos)

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

y_test = np.array(y_test)

print("="*50)
print("РЕЗУЛЬТАТЫ КЛАССИФИКАЦИИ")
print("="*50)

accuracy = (y_pred == y_test).sum() / len(x_test)
print(f"Точность (Accuracy): {accuracy:.4f}")
print(f"Всего тестовых образцов: {len(x_test)}")
print(f"Правильно предсказано: {(y_pred == y_test).sum()}")
print(f"Неправильно предсказано: {(y_pred != y_test).sum()}")

print("\nМатрица ошибок (Confusion Matrix):")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba, pos_label=1)
auc = roc_auc_score(y_test, y_pred_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC-кривая (AUC = {auc:.3f})', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', label='Случайный классификатор')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC-кривая')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

prec_pos, rec_pos, _ = precision_recall_curve(y_test, y_pred_proba, pos_label=1)

y_test_neg = (y_test == -1).astype(int)
y_pred_proba_neg = 1 - y_pred_proba
prec_neg, rec_neg, _ = precision_recall_curve(y_test_neg, y_pred_proba_neg)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(rec_pos, prec_pos, 'b-', linewidth=2)
plt.title('PR-кривая для класса 1')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(rec_neg, prec_neg, 'r-', linewidth=2)
plt.title('PR-кривая для класса -1')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

print("\n" + "="*50)
print("ВЫВОД О КАЧЕСТВЕ КЛАССИФИКАТОРА")
print("="*50)

if auc >= 0.9:
    quality = "отличное"
elif auc >= 0.8:
    quality = "хорошее"
elif auc >= 0.7:
    quality = "среднее"
else:
    quality = "плохое"

print(f"AUC-ROC = {auc:.4f} — это {quality} качество разделения классов")

if auc > 0.95:
    print("Классификатор является «хорошим», так как AUC-ROC близок к 1.")
    print("Классы хорошо разделимы благодаря выбранным параметрам распределений.")
else:
    print("Классификатор не является идеальным, возможны улучшения.")