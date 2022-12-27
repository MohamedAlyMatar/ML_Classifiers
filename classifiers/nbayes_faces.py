from starter_code import *
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import CategoricalNB
from sklearn.naive_bayes import ComplementNB
from sklearn import metrics
# faces data set
x_train = read_lines("../data/facedata/facedatatrain", 70)
y_train = read_labels("../data/facedata/facedatatrainlabels")

x_test = read_lines("../data/facedata/facedatatest", 70)
y_test = read_labels("../data/facedata/facedatatestlabels")

# we need to flat our array from 3D to 2D because the model accepts only 2D/1D
flat_x_train = x_train.reshape(451, 4200)
print(flat_x_train.shape)
flat_x_test = x_test.reshape(150, 4200)
print(flat_x_test.shape)

#create a gaussian model and fit
model = GaussianNB()
model.fit(flat_x_train, y_train)
y_predict = model.predict(flat_x_test)

#Model Accuracy
print("Accuracy:", metrics.accuracy_score(y_test, y_predict))

#visualize first 9
visualize(x_test, y_predict, 9, 70, 60)


# GNB=GaussianNB()
# BNB = BernoulliNB()
# MNB = MultinomialNB()
# ComNB = ComplementNB()
# CatNB = CategoricalNB()



# # Compare between models
# accuracy=[]
# predictions=[]
# for model in [GNB,BNB,MNB,ComNB,CatNB]:
#     model.fit(flat_x_train,y_train)
#     Y_predict=model.predict(flat_x_test)
#     accuracy.append(metrics.accuracy_score(y_test,Y_predict))
#
# models=["Gaussian","Bernoulli","Multinomial","Complement","Categorical"]
# plt.plot(models,accuracy)
# plt.title("Naive Bayes Model Comparison")
# plt.show()