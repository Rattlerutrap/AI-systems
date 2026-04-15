from sklearn.naive_bayes import GaussianNB 

def toNums(field):
    temp = []
    for tile in field:
        if (tile == 'x'):
            temp.append(1)
        elif (tile == 'o'):
            temp.append(0)
        else:
            temp.append(-1)
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




gnb = GaussianNB() 

for i in range(1, len(data)):
    x_train = data[:i]
    y_train = target[:i]

    x_test = data[i:]
    y_test = target[i:]

    gnb.fit(x_train, y_train)
    y_pred = gnb.predict(x_test) 
    print(f"Train size: {len(x_train)} Test size: {len(y_test)}, Correct: {(y_pred == y_test).sum()}, Accuracy: {(y_pred == y_test).sum()/len(y_test)}")
