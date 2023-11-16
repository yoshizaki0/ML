from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
iris = datasets.load_iris()
x, y = iris.data, iris.target
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)
model = svm.LinearSVC()
model.fit(x_train, y_train)
pred = model.predict(x_test)
print("SVC score is "+str(accuracy_score(y_test, pred)*100)+"%")
randomforest = RandomForestClassifier()
randomforest.fit(x_train, y_train)
pred = randomforest.predict(x_test)
print("Randomforest score is "+str(accuracy_score(y_test, pred)*100)+"%")
