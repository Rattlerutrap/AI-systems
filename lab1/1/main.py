from sklearn import datasets 
import os
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
 
gnb.fit(data, target) 
 
y_pred = gnb.predict(data) 
print("Number of mislabeled points out of a total %d points : %d" % (len(data), (target != y_pred).sum())) 
