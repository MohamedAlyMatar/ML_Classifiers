from starter_code import *
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

# # digits data set
# x_train = read_lines("../data/digitdata/trainingimages", 28)
# y_train = read_labels("../data/digitdata/traininglabels")
#
# x_test = read_lines("../data/digitdata/testimages", 28)
# y_test = read_labels("../data/digitdata/testlabels")
#
# # we need to flat our array from 3D to 2D because the model accepts only 2D/1D
# flat_x_train = x_train.reshape(5000, 784)
# print(flat_x_train.shape)
# flat_x_test = x_test.reshape(1000, 784)
# print(flat_x_test.shape)
#
# # create a gaussian model and fit
# model = GaussianNB()
# model.fit(flat_x_train, y_train)
# y_predict = model.predict(flat_x_test)
#
# # Model Accuracy
# print("Accuracy:", metrics.accuracy_score(y_test, y_predict))
#
# # visualize first 9
# visualize(x_train, y_train, 9, 28, 28)

#######################################################################################################################

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

# create a gaussian model and fit
model = GaussianNB()
model.fit(flat_x_train, y_train)
y_predict = model.predict(flat_x_test)

# Model Accuracy
print("Accuracy:", metrics.accuracy_score(y_test, y_predict))

# visualize first 9
visualize(x_train, y_train, 9, 70, 60)