from sources.starter_code import *

# Sklearn modules & classes
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
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

sc = StandardScaler()
sc.fit(flat_x_train)
X_train_std = sc.transform(flat_x_train)
X_test_std = sc.transform(flat_x_test)

# Instantiate the Support Vector Classifier (SVC)
# C: Regularization parameter. The strength of the regularization is inversely proportional to C.
# kernel: Specifies the kernel type to be used in the algorithm.
# kernel values: {‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}
# random_state: Controls the pseudo random number generation for shuffling the data for probability estimates.
svc = SVC(C=10.0, random_state=1, kernel="rbf")

# Fit the model
svc.fit(X_train_std, y_train)
# Make the predictions
y_predict = svc.predict(X_test_std)
# Measure Accuracy
print("Accuracy score %.3f" % metrics.accuracy_score(y_test, y_predict))

# visualize first 9
visualize(x_test, y_predict, 9, 70, 60)
