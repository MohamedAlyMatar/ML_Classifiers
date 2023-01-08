import matplotlib.pyplot as plt

from sources.starter_code import *
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import CategoricalNB
from sklearn.naive_bayes import ComplementNB
from sklearn import metrics

# digits data set
x_train = read_lines("../data/digitdata/trainingimages", 28)
y_train = read_labels("../data/digitdata/traininglabels")

x_test = read_lines("../data/digitdata/testimages", 28)
y_test = read_labels("../data/digitdata/testlabels")

# we need to flat our array from 3D to 2D because the model accepts only 2D/1D
flat_x_train = x_train.reshape(5000, 784)
print(flat_x_train.shape)
flat_x_test = x_test.reshape(1000, 784)
print(flat_x_test.shape)

#create a gaussian model and fit
GNB = GaussianNB()
GNB.fit(flat_x_train, y_train)
y_predict = GNB.predict(flat_x_test)

#Model Accuracy
print("Accuracy:", metrics.accuracy_score(y_test, y_predict))

#visualize first 9
visualize(x_test, y_predict, 9, 28, 28)

# Compare between models
# GNB=GaussianNB()
# BNB = BernoulliNB()
# MNB = MultinomialNB()
# ComNB = ComplementNB()
#
#
# accuracy=[]
# predictions=[]
# for model in [GNB,BNB,MNB,ComNB]:
#     model.fit(flat_x_train,y_train)
#     Y_predict=model.predict(flat_x_test)
#     accuracy.append(metrics.accuracy_score(y_test,Y_predict))
#
# models=["Gaussian","Bernoulli","Multinomial","Complement"]
# plt.plot(models,accuracy)
# plt.title("Naive Bayes Model Comparison")
# plt.show()






