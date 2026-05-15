import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import random

import sys
sys.path.append(r'F:\Системы ИИ')
import utility

data_train = utility.importCSV('titanic_train.csv', 1, 0, ',')
data_train = np.array(data_train)

data_test = utility.importCSV('titanic_test.csv', 1, 0, ',')
data_test = np.array(data_test)


x_train = []
x_test = []
y_train = []

# PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
for i in range(len(data_train)):
    row = data_train[i]
    passId = int(row[0])
    survived = True if row[1] == '1' else False
    pClass = int(row[2])
    name = row[3]
    sex = 1 if row[4] == 'male' else 0
    age = float(row[5]) if row[5] != '' else -1
    sib = int(row[6]) if row[6] != '' else -1
    parch = int(row[7]) if row[7] != '' else -1
    ticket = row[8]
    fare = float(row[9]) if row[9] != '' else -1
    cabin = row[10]
    port = 0 if row[11] == 'C' else (1 if row[11] == 'S' else 2)
    y_train.append(survived)
    x_train.append([
        pClass,
        sex,
        age,
        sib,
        parch,
        fare,
        port
    ])

# PassengerId,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
for i in range(len(data_test)):
    row = data_test[i]
    passId = int(row[0])
    pClass = int(row[1])
    name = row[2]
    sex = 1 if row[3] == 'male' else 0
    age = float(row[4]) if row[4] != '' else -1
    sib = int(row[5]) if row[5] != '' else -1
    parch = int(row[6]) if row[6] != '' else -1
    ticket = row[7]
    fare = float(row[8]) if row[8] != '' else -1
    cabin = row[9]
    port = 0 if row[10] == 'C' else (1 if row[10] == 'S' else 2)
    x_test.append([
        pClass,
        sex,
        age,
        sib,
        parch,
        fare,
        port
    ])


model1 = DecisionTreeClassifier(max_depth=10)
model2 = SVC(probability=True)

model1.fit(x_train, y_train)
model2.fit(x_train, y_train)
 
# Получаем предсказания моделей
pred1_train = model1.predict(x_train)
pred2_train = model2.predict(x_train)

# Создаем признаки для мета-модели
X_meta_train = np.column_stack([pred1_train, pred2_train])

# Обучаем мета-модель (стеккинг)
meta_model = LogisticRegression()
meta_model.fit(X_meta_train, y_train)

# Предсказания для теста
pred1_test = model1.predict(x_test)
pred2_test = model2.predict(x_test)
X_meta_test = np.column_stack([pred1_test, pred2_test])


# Финальные предсказания
final_predictions = meta_model.predict(X_meta_test)


# Сохраняем результат
passenger_ids = data_test[:, 0]
output = np.column_stack([passenger_ids, final_predictions])
np.savetxt('submission.csv', output, delimiter=',', fmt='%s', header='PassengerId,Survived', comments='')


result = []
for i in range(len(x_test)):
    result.append([
        passenger_ids[i], #pId
        x_test[i][0], #pClass
        x_test[i][1], #sex
        x_test[i][2], #age
        x_test[i][3], #siblings
        x_test[i][4], #parents/children
        x_test[i][6], #embarked
        final_predictions[i]
    ]
    )

class_surv = [0]*3
class_dead = [0]*3

sex_surv = [0]*2
sex_dead = [0]*2

ages_surv = [0]*100
ages_dead = [0]*100

sibl_surv = [0]*10
sibl_dead = [0]*10

parch_surv = [0]*10
parch_dead = [0]*10

port_surv = [0]*3
port_dead = [0]*3

id_surv = []
id_dead = []

for i in result:
    if i[-1] == 1:
        id_surv.append(i[0])
        class_surv[round(i[1] - 1)] += 1
        sex_surv[round(i[2])] += 1
        ages_surv[round(i[3])] += 1
        sibl_surv[round(i[4])] += 1
        parch_surv[round(i[5])] += 1
        port_surv[round(i[6])] += 1
    else:
        id_dead.append(i[0])
        class_dead[round(i[1]- 1)] += 1
        sex_dead[round(i[2])] += 1
        ages_dead[round(i[3])] += 1
        sibl_dead[round(i[4])] += 1
        parch_dead[round(i[5])] += 1
        port_dead[round(i[6])] += 1

print(f'Доля выживших мужчин: {sex_surv[1] / (sex_surv[1] + sex_dead[1])}')
print(f'Доля выживших женщин: {sex_surv[0] / (sex_surv[0] + sex_dead[0])}')

print(f'Доля выживших 1 класса: {class_surv[0] / (class_surv[0] + class_dead[0])}')
print(f'Доля выживших 2 класса: {class_surv[1] / (class_surv[1] + class_dead[1])}')
print(f'Доля выживших 3 класса: {class_surv[2] / (class_surv[2] + class_dead[2])}')

print(f'Доля выживших с порта Cherbourg: {port_surv[0] / (port_surv[0] + port_dead[0])}')
print(f'Доля выживших с порта Southampton: {port_surv[1] / (port_surv[1] + port_dead[1])}')
print(f'Доля выживших с порта Queenstown: {port_surv[2] / (port_surv[2] + port_dead[2])}')

plt.figure(figsize=(12, 6))
plt.plot(ages_surv[:-1], color='b', linewidth=2, label='Выжившие')
plt.plot(ages_dead[:-1], color='r', linewidth=2, label='Погибшие')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlabel('Возраст')
plt.ylabel('Человек')
plt.title('Зависимость выживаемости от возраста')
plt.savefig('age')
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(sibl_surv[:-1], color='b', linewidth=2, label='Выжившие')
plt.plot(sibl_dead[:-1], color='r', linewidth=2, label='Погибшие')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlabel('Количество родственников')
plt.ylabel('Человек')
plt.title('Зависимость выживаемости от количества родственников')
plt.savefig('sib')
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(parch_surv[:-1], color='b', linewidth=2, label='Выжившие')
plt.plot(parch_dead[:-1], color='r', linewidth=2, label='Погибшие')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlabel('Количество родителей/детей')
plt.ylabel('Человек')
plt.title('Зависимость выживаемости от количества родителей/детей')
plt.savefig('parch')
plt.show()


        