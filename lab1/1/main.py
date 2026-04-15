from sklearn.naive_bayes import GaussianNB, CategoricalNB
import matplotlib.pyplot as plt
import random

def toNums(field):
    temp = []
    for tile in field:
        if (tile == 'x'):
            temp.append(2)
        elif (tile == 'o'):
            temp.append(1)
        else:
            temp.append(0)
    return temp

def toNumRes(res):
    return 1 if res == 'positive\n' else 0


data = []
target = []

with open('tic_tac_toe.txt', 'r', encoding='utf-8') as file:
    line = file.readline()
    while line:
        line = line.split(',')
        target.append(toNumRes(line.pop()))
        data.append(toNums(line))
        line = file.readline()




gnb = CategoricalNB(min_categories=3) 

trainToTestingRatio = []
accuracy = []

for i in range(1, len(data)):
    x_train = data[:i]
    y_train = target[:i]

    x_test = data[i:]
    y_test = target[i:]

    gnb.fit(x_train, y_train)
    y_pred = gnb.predict(x_test) 

    trainToTestingRatio.append(len(x_train)/len(x_test))
    accuracy.append((y_pred == y_test).sum()/len(y_test))


plt.figure(figsize=(10, 6))
plt.plot(trainToTestingRatio, accuracy, 'b-', linewidth=2)
plt.xlabel('Train/Test Ratio')
plt.ylabel('Accuracy')
plt.title('Зависимость точности от соотношения обучающей и тестовой выборок')
plt.grid(True)
plt.show()


trainToTestingRatio = []
accuracy = []

for i in range(len(data)):
    data[i].append(target[i])
random.shuffle(data)
target = []
for i in data:
    target.append(i[-1])


for i in range(1, len(data)):
    x_train = data[:i]
    y_train = target[:i]

    x_test = data[i:]
    y_test = target[i:]

    gnb.fit(x_train, y_train)
    y_pred = gnb.predict(x_test) 

    trainToTestingRatio.append(len(x_train)/len(x_test))
    accuracy.append((y_pred == y_test).sum()/len(y_test))
    print(f"Train size: {len(x_train)} Test size: {len(y_test)}, Correct: {(y_pred == y_test).sum()}, Accuracy: {(y_pred == y_test).sum()/len(y_test)}")


plt.figure(figsize=(10, 6))
plt.plot(trainToTestingRatio, accuracy, 'b-', linewidth=2)
plt.xlabel('Train/Test Ratio')
plt.ylabel('Accuracy')
plt.title('Зависимость точности от соотношения обучающей и тестовой выборок')
plt.grid(True)
plt.show()