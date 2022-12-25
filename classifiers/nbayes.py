from starter_code import *
from sklearn.naive_bayes import GaussianNB
# Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
import matplotlib.pyplot as plt
model = GaussianNB()

X_train=read_lines2("../data/digitdata/trainingimages",5000)
Y_train=read_labels("../data/digitdata/traininglabels")

X_test=read_lines2("../data/digitdata/testimages",1000)
Y_test=read_labels("../data/digitdata/testlabels")

model.fit(X_train,Y_train)
y_predict = model.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:", metrics.accuracy_score(Y_test, y_predict))