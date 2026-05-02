print("\n" + "="*80)
print("ПУНКТ 6: Bank Scoring")
print("="*80)

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utility
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                             roc_auc_score, roc_curve, f1_score, precision_score, 
                             recall_score, precision_recall_curve)
import matplotlib.pyplot as plt
import numpy as np
import random

data_test = utility.importCSV('bank_scoring_test.csv', 1, delimiter='\t')
data_train = utility.importCSV('bank_scoring_train.csv', 1, delimiter='\t')

x_test = []
y_test = []
x_train = []
y_train = []

for i in data_train:
    x_train.append([float(val) for val in i[1:]])
    y_train.append(float(i[0]))

for i in data_test:
    x_test.append([float(val) for val in i[1:]])
    y_test.append(float(i[0]))

print(f"Размер train: {len(x_train)}, test: {len(x_test)}")
print(f"Распределение классов в train: {np.bincount(y_train)}")
print(f"Распределение классов в test: {np.bincount(y_test)}")

class_ratio_train = sum(y_train) / len(y_train)
class_ratio_test = sum(y_test) / len(y_test)
print(f"Доля дефолтов (класс 1) в train: {class_ratio_train:.4f} ({class_ratio_train*100:.2f}%)")
print(f"Доля дефолтов (класс 1) в test: {class_ratio_test:.4f} ({class_ratio_test*100:.2f}%)")

print("\n=== Классификатор 1: Decision Tree ===")

dt = DecisionTreeClassifier(random_state=42, max_depth=5, criterion='entropy', class_weight='balanced')
dt.fit(x_train, y_train)

y_pred_dt = dt.predict(x_test)
y_pred_proba_dt = dt.predict_proba(x_test)[:, 1]

print(f"Accuracy: {accuracy_score(y_test, y_pred_dt):.4f}")
print(f"AUC-ROC: {roc_auc_score(y_test, y_pred_proba_dt):.4f}")
print(f"F1-score: {f1_score(y_test, y_pred_dt):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_dt):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_dt):.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred_dt, target_names=['good', 'default']))

print("Confusion Matrix:")
cm_dt = confusion_matrix(y_test, y_pred_dt)
print(cm_dt)

print("\n=== Классификатор 2: Gaussian Naive Bayes ===")

gnb = GaussianNB()
gnb.fit(x_train, y_train)

y_pred_gnb = gnb.predict(x_test)
y_pred_proba_gnb = gnb.predict_proba(x_test)[:, 1]

print(f"Accuracy: {accuracy_score(y_test, y_pred_gnb):.4f}")
print(f"AUC-ROC: {roc_auc_score(y_test, y_pred_proba_gnb):.4f}")
print(f"F1-score: {f1_score(y_test, y_pred_gnb):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_gnb):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_gnb):.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred_gnb, target_names=['good', 'default']))

print("Confusion Matrix:")
cm_gnb = confusion_matrix(y_test, y_pred_gnb)
print(cm_gnb)

plt.figure(figsize=(10, 8))

fpr_dt, tpr_dt, _ = roc_curve(y_test, y_pred_proba_dt)
auc_dt = roc_auc_score(y_test, y_pred_proba_dt)
plt.plot(fpr_dt, tpr_dt, label=f'Decision Tree (AUC = {auc_dt:.3f})')

fpr_gnb, tpr_gnb, _ = roc_curve(y_test, y_pred_proba_gnb)
auc_gnb = roc_auc_score(y_test, y_pred_proba_gnb)
plt.plot(fpr_gnb, tpr_gnb, label=f'GaussianNB (AUC = {auc_gnb:.3f})')

plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.5)')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-кривые для кредитного скоринга')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.savefig('bank_scoring_roc_curves.png', dpi=150, bbox_inches='tight')
plt.show()

plt.figure(figsize=(10, 8))

precision_dt, recall_dt, _ = precision_recall_curve(y_test, y_pred_proba_dt)
plt.plot(recall_dt, precision_dt, label='Decision Tree')

precision_gnb, recall_gnb, _ = precision_recall_curve(y_test, y_pred_proba_gnb)
plt.plot(recall_gnb, precision_gnb, label='GaussianNB')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('PR-кривые для кредитного скоринга')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('bank_scoring_pr_curves.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "="*80)
print("ВЫВОДЫ ПО СРАВНЕНИЮ КЛАССИФИКАТОРОВ")
print("="*80)

metrics = {
    'Decision Tree': {
        'AUC-ROC': roc_auc_score(y_test, y_pred_proba_dt),
        'F1-score': f1_score(y_test, y_pred_dt),
        'Recall': recall_score(y_test, y_pred_dt),
        'Precision': precision_score(y_test, y_pred_dt)
    },
    'GaussianNB': {
        'AUC-ROC': roc_auc_score(y_test, y_pred_proba_gnb),
        'F1-score': f1_score(y_test, y_pred_gnb),
        'Recall': recall_score(y_test, y_pred_gnb),
        'Precision': precision_score(y_test, y_pred_gnb)
    }
}

for name, metric in metrics.items():
    print(f"\n{name}:")
    print(f"  AUC-ROC: {metric['AUC-ROC']:.4f}")
    print(f"  F1-score: {metric['F1-score']:.4f}")
    print(f"  Recall:   {metric['Recall']:.4f}")
    print(f"  Precision:{metric['Precision']:.4f}")

best_model = max(metrics, key=lambda x: metrics[x]['AUC-ROC'])
print(f"\nЛучший классификатор по AUC-ROC: {best_model}")