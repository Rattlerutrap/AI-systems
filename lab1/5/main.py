import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utility

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt   
import numpy as np
import random


def punktA():
    data = utility.importCSV('glass.csv', 1, 1)
    x = [[float(item) for item in row] for row in data]

    random.shuffle(x)
    y = []
    for i in x:
        y.append(i[-1])
        i.pop()

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
    print(f"Точность entropy дерева: {acc_base:.4f}")

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
    print(f"Точность max_depth=3 дерева: {acc_base:.4f}")

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
    print(f"Точность max_depth=5 дерева: {acc_base:.4f}")

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
    print(f"Точность max_depth=10 дерева: {acc_base:.4f}")

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

def punktB():
    data = utility.importCSV('spam7.csv', 1)
    random.shuffle(data)

    x = []
    y = []
    for i in data:
        x.append([float(val) for val in i[:-1]])
        if i[-1] == 'n':
            y.append(0)
        else:
            y.append(1)

    split_idx = int(len(x) * 0.8)
    x_train = x[:split_idx]
    y_train = y[:split_idx]
    x_test = x[split_idx:]
    y_test = y[split_idx:]

    print(f"Размер train: {len(x_train)}, test: {len(x_test)}")
    print(f"Распределение классов в train: {np.bincount(y_train)}")
    print(f"Распределение классов в test: {np.bincount(y_test)}")

    print("\n=== Поиск оптимальных параметров через GridSearchCV ===")
    
    param_grid = {
        'max_depth': [3, 5, 7, 10, 15, None],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 4, 8],
        'criterion': ['gini', 'entropy']
    }
    
    grid_search = GridSearchCV(
        DecisionTreeClassifier(random_state=42),
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(x_train, y_train)
    
    print(f"\nЛучшие параметры: {grid_search.best_params_}")
    print(f"Лучшая кросс-валидационная точность: {grid_search.best_score_:.4f}")
    
    best_tree = grid_search.best_estimator_
    
    y_pred = best_tree.predict(x_test)
    y_pred_proba = best_tree.predict_proba(x_test)[:, 1]
    
    print("\n=== Оценка качества на тестовой выборке ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"F1-score: {f1_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['no (n)', 'yes (y)']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    feature_names = ["crl.tot", "dollar", "bang", "money", "n000", "make"]
    importances = best_tree.feature_importances_
    
    print("\n=== Важность признаков ===")
    for name, imp in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True):
        print(f"  {name}: {imp:.4f}")
    
    plt.figure(figsize=(10, 6))
    plt.barh(feature_names, importances)
    plt.xlabel("Важность")
    plt.title("Важность признаков для дерева решений (spam7)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('feature_importance_spam7.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    plt.figure(figsize=(30, 20))
    plot_tree(best_tree, 
              feature_names=feature_names, 
              class_names=['no', 'yes'], 
              filled=True, 
              fontsize=8,
              proportion=True,
              rounded=True)
    plt.title("Оптимальное дерево решений для spam7", fontsize=16)
    plt.tight_layout()
    plt.savefig('decision_tree_spam7_optimal.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return best_tree, importances 

punktB()