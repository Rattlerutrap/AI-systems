from sklearn import datasets 
from sklearn.naive_bayes import GaussianNB 

class DataEl:
    def __init__(self):
        pass

iris = datasets.load_iris() 
 
gnb = GaussianNB() 
 
gnb.fit(iris.data, iris.target) 
 
y_pred = gnb.predict(iris.data) 
print("Number of mislabeled points out of a total %d points : %d" % (iris.data.shape[0],(iris.target != y_pred).sum())) 
