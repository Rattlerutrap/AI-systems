import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utility

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB, CategoricalNB
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

clf_base = DecisionTreeClassifier(random_state=42, max_depth=5, criterion='entropy')
clf_base.fit(x_train, y_train)
y_pred_tree = clf_base.predict(x_test)
acc_tree = accuracy_score(y_test, y_pred_tree)
print(f"Точность дерева: {acc_tree:.4f}")

gnb = GaussianNB()
gnb.fit(x_train, y_train)
y_gauss = gnb.predict(x_test) 
acc_gauss = accuracy_score(y_test, y_gauss)
print(f"Точность Байеса: {acc_gauss:.4f}")