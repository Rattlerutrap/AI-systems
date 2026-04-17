import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utility

data = utility.importCSV('glass.csv', 1, 1)

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt   
import numpy as np
import random

x = [[float(item) for item in row] for row in data]

random.shuffle(x)
y = []
for i in x:
    y.append(i[-1])
    i.pop()
    
scaler = StandardScaler()
data_normalized = scaler.fit_transform(x)

x_train = x[:150]
y_train = y[:150]

x_test = x[150:]
y_test = y[150:]

feature_names = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']

unique_classes = sorted(set(y))


clf_base = DecisionTreeClassifier(random_state=42)
clf_base.fit(x_train, y_train)

y_pred_base = clf_base.predict(x_test)
acc_base = accuracy_score(y_test, y_pred_base)
print(f"Точность базового дерева: {acc_base:.4f}")

plt.figure(figsize=(40, 25))
plot_tree(clf_base, 
          feature_names=feature_names, 
          class_names=[str(i) for i in unique_classes], 
          filled=True, 
          fontsize=7,
          proportion=True,
          rounded=True)
plt.title("Полное дерево решений для Glass Dataset (без обрезки)", fontsize=16)
plt.tight_layout()
plt.savefig('decision_tree.png', dpi=300, bbox_inches='tight') 


clf_entropy = DecisionTreeClassifier(criterion='entropy', random_state=42)
clf_entropy.fit(x_train, y_train)

y_pred_base = clf_entropy.predict(x_test)
acc_base = accuracy_score(y_test, y_pred_base)
print(f"Точность базового дерева: {acc_base:.4f}")

plt.figure(figsize=(40, 25))
plot_tree(clf_entropy, 
          feature_names=feature_names, 
          class_names=[str(i) for i in unique_classes], 
          filled=True, 
          fontsize=7,
          proportion=True,
          rounded=True)
plt.title("Полное дерево решений для clf_entropy", fontsize=16)
plt.tight_layout()
plt.savefig('decision_tree_entropy.png', dpi=300, bbox_inches='tight') 


clf_depth3 = DecisionTreeClassifier(random_state=42, max_depth=3)
clf_depth3.fit(x_train, y_train)

y_pred_base = clf_depth3.predict(x_test)
acc_base = accuracy_score(y_test, y_pred_base)
print(f"Точность базового дерева: {acc_base:.4f}")

plt.figure(figsize=(40, 25))
plot_tree(clf_depth3, 
          feature_names=feature_names, 
          class_names=[str(i) for i in unique_classes], 
          filled=True, 
          fontsize=7,
          proportion=True,
          rounded=True)
plt.title("Полное дерево решений для clf_depth3", fontsize=16)
plt.tight_layout()
plt.savefig('decision_tree_depth3.png', dpi=300, bbox_inches='tight') 


clf_depth5 = DecisionTreeClassifier(random_state=42, max_depth=5)
clf_depth5.fit(x_train, y_train)

y_pred_base = clf_depth5.predict(x_test)
acc_base = accuracy_score(y_test, y_pred_base)
print(f"Точность базового дерева: {acc_base:.4f}")

plt.figure(figsize=(40, 25))
plot_tree(clf_depth5, 
          feature_names=feature_names, 
          class_names=[str(i) for i in unique_classes], 
          filled=True, 
          fontsize=7,
          proportion=True,
          rounded=True)
plt.title("Полное дерево решений для clf_depth5", fontsize=16)
plt.tight_layout()
plt.savefig('decision_tree_depth5.png', dpi=300, bbox_inches='tight') 


clf_depth5 = DecisionTreeClassifier(random_state=42, max_depth=10)
clf_depth5.fit(x_train, y_train)

y_pred_base = clf_depth5.predict(x_test)
acc_base = accuracy_score(y_test, y_pred_base)
print(f"Точность базового дерева: {acc_base:.4f}")

plt.figure(figsize=(40, 25))
plot_tree(clf_depth5, 
          feature_names=feature_names, 
          class_names=[str(i) for i in unique_classes], 
          filled=True, 
          fontsize=7,
          proportion=True,
          rounded=True)
plt.title("Полное дерево решений для clf_depth5", fontsize=16)
plt.tight_layout()
plt.savefig('decision_tree_depth10.png', dpi=300, bbox_inches='tight') 
