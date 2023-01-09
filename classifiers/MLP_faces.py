################################### imports ##################################################
from starter_code import *
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


################################# digits data set ############################################

###Data Preprocessing###

x_train = read_lines("../data/facedata/facedatatrain", 70)
y_train = read_labels("../data/facedata/facedatatrainlabels")

x_test = read_lines("../data/facedata/facedatatest", 70)
y_test = read_labels("../data/facedata/facedatatestlabels")

# we need to flat our array from 3D to 2D because the model accepts only 2D/1D

flat_x_train = x_train.reshape(451, 4200)
flat_x_test = x_test.reshape(150, 4200)

clf = MLPClassifier(hidden_layer_sizes=(7,6),
                    random_state=5,
                    verbose=True,
                    learning_rate_init=0.091,
                    activation='identity')

# Fit data onto the model
clf.fit(flat_x_train,y_train)
print(clf)

# Testing the model
y_pred=clf.predict(flat_x_test)

test_accuracy = clf.score(flat_x_test, y_test)

print("Accuracy is")
print(test_accuracy)

visualize(x_test, y_pred, 9, 70, 60)



