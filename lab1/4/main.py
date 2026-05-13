import sys
import os
sys.path.append(r'F:\Системы ИИ')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utility

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
from matplotlib.colors import ListedColormap


counter = 1

def plot_decision_boundary(model, X, y, title, support_vectors=None):
    """
    Визуализация границ решений SVM
    """

    plt.figure(figsize=(10, 8))
    
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    cmap_light = ListedColormap(["#680000", '#AAFFAA'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00'])
    
    plt.contourf(xx, yy, Z, cmap=cmap_light)
    
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, 
                         edgecolor='black', s=50, alpha=0.8)
    
    if support_vectors is not None:
        plt.scatter(support_vectors[:, 0], support_vectors[:, 1], 
           marker='s', s=200, facecolors='none', 
           edgecolors='blue', linewidth=2)
    
    if model.kernel == 'linear':
        w = model.coef_[0]
        b = model.intercept_[0]
        
        x_line = np.array([x_min, x_max])
        y_line = -(w[0] * x_line + b) / w[1]
        plt.plot(x_line, y_line, 'k-', linewidth=2, label='Разделяющая линия')
        
        y_margin_up = -(w[0] * x_line + b - 1) / w[1]
        y_margin_down = -(w[0] * x_line + b + 1) / w[1]
        plt.plot(x_line, y_margin_up, 'k--', linewidth=1, alpha=0.7)
        plt.plot(x_line, y_margin_down, 'k--', linewidth=1, alpha=0.7)
    
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    plt.xlabel('Признак 1')
    plt.ylabel('Признак 2')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    global counter
    plt.savefig(f'a{counter}')
    counter += 1
    # plt.show()
    

def punktA():
    data_train= utility.importTxt('svmdata_a.txt', 1, 1, '\t')
    data_test = utility.importTxt('svmdata_a_test.txt', 1, 1, '\t')

    X_train = np.array([[float(x[0]), float(x[1])] for x in data_train])
    Y_train = np.array([1 if x[2] == 'green' else 0 for x in data_train])

    X_test = np.array([[float(x[0]), float(x[1])] for x in data_test])
    Y_test = np.array([1 if x[2] == 'green' else 0 for x in data_test])

    svm_linear = SVC(kernel='linear', random_state=42)
    svm_linear.fit(X_train, Y_train)

    y_train_pred = svm_linear.predict(X_train)
    y_test_pred = svm_linear.predict(X_test)

    train_accuracy = accuracy_score(Y_train, y_train_pred)
    test_accuracy = accuracy_score(Y_test, y_test_pred)

    print("=== SVM с линейным ядром ===")
    print(f"Точность на обучающей выборке: {train_accuracy:.4f}")
    print(f"Точность на тестовой выборке: {test_accuracy:.4f}")
    print(f"Количество опорных векторов: {len(svm_linear.support_vectors_)}")

    cm_train = confusion_matrix(Y_train, y_train_pred)
    print("\nМатрица ошибок (обучение):")
    print("          Предсказано")
    print("          Класс 0  Класс 1")
    print(f"Класс 0   {cm_train[0,0]:6d}  {cm_train[0,1]:6d}")
    print(f"Класс 1   {cm_train[1,0]:6d}  {cm_train[1,1]:6d}")

    cm_test = confusion_matrix(Y_test, y_test_pred)
    print("\nМатрица ошибок (тест):")
    print("          Предсказано")
    print("          Класс 0  Класс 1")
    print(f"Класс 0   {cm_test[0,0]:6d}  {cm_test[0,1]:6d}")
    print(f"Класс 1   {cm_test[1,0]:6d}  {cm_test[1,1]:6d}")


    plot_decision_boundary(
        svm_linear, 
        X_train, 
        Y_train, 
        f'SVM с линейным ядром (обучение)\nОпорных векторов: {len(svm_linear.support_vectors_)}',
        support_vectors=svm_linear.support_vectors_
    )

    plot_decision_boundary(
        svm_linear, 
        X_test, 
        Y_test, 
        f'SVM с линейным ядром (тест)\nОпорных векторов: {len(svm_linear.support_vectors_)}',
        support_vectors=svm_linear.support_vectors_
    )

punktA()

def punktB():
    data_train= utility.importTxt('svmdata_b.txt', 1, 1, '\t')
    data_test = utility.importTxt('svmdata_b_test.txt', 1, 1, '\t')

    X_train = np.array([[float(x[0]), float(x[1])] for x in data_train])
    Y_train = np.array([1 if x[2] == 'green' else 0 for x in data_train])

    X_test = np.array([[float(x[0]), float(x[1])] for x in data_test])
    Y_test = np.array([1 if x[2] == 'green' else 0 for x in data_test])


    #на 500 отсутствуют ошибки на обучающей выборке, но появляются на тестовой
    svm_linear = SVC(kernel='linear', random_state=42)
    svm_linear.fit(X_train, Y_train)

    y_train_pred = svm_linear.predict(X_train)
    y_test_pred = svm_linear.predict(X_test)

    train_accuracy = accuracy_score(Y_train, y_train_pred)
    test_accuracy = accuracy_score(Y_test, y_test_pred)

    print("=== SVM с линейным ядром ===")
    print(f"Точность на обучающей выборке: {train_accuracy:.4f}")
    print(f"Точность на тестовой выборке: {test_accuracy:.4f}")
    print(f"Количество опорных векторов: {len(svm_linear.support_vectors_)}")

    cm_train = confusion_matrix(Y_train, y_train_pred)
    print("\nМатрица ошибок (обучение):")
    print("          Предсказано")
    print("          Класс 0  Класс 1")
    print(f"Класс 0   {cm_train[0,0]:6d}  {cm_train[0,1]:6d}")
    print(f"Класс 1   {cm_train[1,0]:6d}  {cm_train[1,1]:6d}")

    cm_test = confusion_matrix(Y_test, y_test_pred)
    print("\nМатрица ошибок (тест):")
    print("          Предсказано")
    print("          Класс 0  Класс 1")
    print(f"Класс 0   {cm_test[0,0]:6d}  {cm_test[0,1]:6d}")
    print(f"Класс 1   {cm_test[1,0]:6d}  {cm_test[1,1]:6d}")


    plot_decision_boundary(
        svm_linear, 
        X_train, 
        Y_train, 
        f'SVM с линейным ядром (обучение)\nОпорных векторов: {len(svm_linear.support_vectors_)}',
        support_vectors=svm_linear.support_vectors_
    )

    plot_decision_boundary(
        svm_linear, 
        X_test, 
        Y_test, 
        f'SVM с линейным ядром (тест)\nОпорных векторов: {len(svm_linear.support_vectors_)}',
        support_vectors=svm_linear.support_vectors_
    )

# punktB()

def makeSVMAnalysis(kernel, X_train, Y_train, X_test, Y_test, degree=1, gamma=1):
    if (kernel == 'poly'):
        svm = SVC(kernel=kernel, random_state=42, degree=degree, gamma=gamma)
    elif (kernel == 'linear'):
        svm = SVC(kernel=kernel, random_state=42)
    else:
        svm = SVC(kernel=kernel, random_state=42, gamma=gamma)


    svm.fit(X_train, Y_train)
    y_train_pred = svm.predict(X_train)
    y_test_pred = svm.predict(X_test)

    train_accuracy = accuracy_score(Y_train, y_train_pred)
    test_accuracy = accuracy_score(Y_test, y_test_pred)
    
    if (kernel == 'poly'):
        kernelName = f'{kernel}(degree={degree})'
    else:
        kernelName = f'{kernel}'
    
    if (gamma != 1):
        kernelName += f'(gamma={gamma})'

    print(f"=== SVM с {kernelName} ядром ===")
    # print(f"Точность на обучающей выборке: {train_accuracy:.4f}")
    print(f"Точность на тестовой выборке: {test_accuracy:.4f}")
    print(f"Количество опорных векторов: {len(svm.support_vectors_)}")

    # cm_train = confusion_matrix(Y_train, y_train_pred)
    # print("\nМатрица ошибок (обучение):")
    # print("          Предсказано")
    # print("          Класс 0  Класс 1")
    # print(f"Класс 0   {cm_train[0,0]:6d}  {cm_train[0,1]:6d}")
    # print(f"Класс 1   {cm_train[1,0]:6d}  {cm_train[1,1]:6d}")

    cm_test = confusion_matrix(Y_test, y_test_pred)
    print("\nМатрица ошибок (тест):")
    print("          Предсказано")
    print("          Класс 0  Класс 1")
    print(f"Класс 0   {cm_test[0,0]:6d}  {cm_test[0,1]:6d}")
    print(f"Класс 1   {cm_test[1,0]:6d}  {cm_test[1,1]:6d}")


    # plot_decision_boundary(
    #     svm, 
    #     X_train, 
    #     Y_train, 
    #     f'SVM с {kernelName} ядром (обучение)\nОпорных векторов: {len(svm.support_vectors_)}',
    #     support_vectors=svm.support_vectors_
    # )

    plot_decision_boundary(
        svm, 
        X_test, 
        Y_test, 
        f'SVM с {kernelName} ядром (тест)\nОпорных векторов: {len(svm.support_vectors_)}',
        support_vectors=svm.support_vectors_
    )

def punktC():
    data_train= utility.importTxt('svmdata_c.txt', 1, 1, '\t')
    data_test = utility.importTxt('svmdata_c_test.txt', 1, 1, '\t')

    X_train = np.array([[float(x[0]), float(x[1])] for x in data_train])
    Y_train = np.array([1 if x[2] == 'green' else 0 for x in data_train])

    X_test = np.array([[float(x[0]), float(x[1])] for x in data_test])
    Y_test = np.array([1 if x[2] == 'green' else 0 for x in data_test])

    makeSVMAnalysis('linear', X_train, Y_train, X_test, Y_test)
    for i in range(5):
        makeSVMAnalysis('poly', X_train, Y_train, X_test, Y_test, i+1)
    makeSVMAnalysis('rbf', X_train, Y_train, X_test, Y_test)
    makeSVMAnalysis('sigmoid', X_train, Y_train, X_test, Y_test)

# punktC()

def punktD():
    data_train= utility.importTxt('svmdata_d.txt', 1, 1, '\t')
    data_test = utility.importTxt('svmdata_d_test.txt', 1, 1, '\t')

    X_train = np.array([[float(x[0]), float(x[1])] for x in data_train])
    Y_train = np.array([1 if x[2] == 'green' else 0 for x in data_train])

    X_test = np.array([[float(x[0]), float(x[1])] for x in data_test])
    Y_test = np.array([1 if x[2] == 'green' else 0 for x in data_test])

    makeSVMAnalysis('linear', X_train, Y_train, X_test, Y_test)
    for i in range(5):
        makeSVMAnalysis('poly', X_train, Y_train, X_test, Y_test, i+1)
    makeSVMAnalysis('rbf', X_train, Y_train, X_test, Y_test)
    makeSVMAnalysis('sigmoid', X_train, Y_train, X_test, Y_test)

# punktD()

def punktE():
    data_train= utility.importTxt('svmdata_e.txt', 1, 1, '\t')
    data_test = utility.importTxt('svmdata_e_test.txt', 1, 1, '\t')

    X_train = np.array([[float(x[0]), float(x[1])] for x in data_train])
    Y_train = np.array([1 if x[2] == 'green' else 0 for x in data_train])

    X_test = np.array([[float(x[0]), float(x[1])] for x in data_test])
    Y_test = np.array([1 if x[2] == 'green' else 0 for x in data_test])


    for i in range(5):
        makeSVMAnalysis('poly', X_train, Y_train, X_test, Y_test, i+1, gamma=0.01)
    makeSVMAnalysis('rbf', X_train, Y_train, X_test, Y_test, gamma=0.01)
    makeSVMAnalysis('sigmoid', X_train, Y_train, X_test, Y_test, gamma=0.01)

    for i in range(5):
        makeSVMAnalysis('poly', X_train, Y_train, X_test, Y_test, i+1, gamma=1)
    makeSVMAnalysis('rbf', X_train, Y_train, X_test, Y_test, gamma=1)
    makeSVMAnalysis('sigmoid', X_train, Y_train, X_test, Y_test, gamma=1)

    for i in range(5):
        makeSVMAnalysis('poly', X_train, Y_train, X_test, Y_test, i+1, gamma=100)
    makeSVMAnalysis('rbf', X_train, Y_train, X_test, Y_test, gamma=100)
    makeSVMAnalysis('sigmoid', X_train, Y_train, X_test, Y_test, gamma=100)

# punktE()
