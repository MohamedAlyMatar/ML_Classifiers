################################### imports ##################################################
from starter_code import *
from sklearn.neighbors import KNeighborsClassifier

#################################################faces data set######################################################################



###Data Preprocessing###
face_x_train = read_lines("../data/facedata/facedatatrain", 70)
face_y_train = read_labels("../data/facedata/facedatatrainlabels")

face_x_test = read_lines("../data/facedata/facedatatest", 70)
face_y_test = read_labels("../data/facedata/facedatatestlabels")

# we need to flat our array from 3D to 2D because the model accepts only 2D/1D
flat_face_x_train = face_x_train.reshape(451, 4200)
flat_face_x_test = face_x_test.reshape(150, 4200)

###Applying KNN###

#The number of neighbors is our hyperparamter
face_neighbors = np.arange(2, 15)
face_test_accuracy = np.empty(len(face_neighbors))

#variables needed to compute the k with the highest accuracy
max_accuracy=0
best_k=0

for i, k in enumerate(face_neighbors):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(flat_face_x_train, face_y_train)

    #Compute test data accuracy
    face_test_accuracy[i] = knn.score(flat_face_x_test, face_y_test)
    #Compute the best value of neighbors in the neighbors array
    if(face_test_accuracy[i]>max_accuracy):
        max_accuracy=face_test_accuracy[i]
        best_k=k

#plotting the testing accuracy against the different number of neighbors
plt.plot(face_neighbors, face_test_accuracy, label='Testing dataset Accuracy')
plt.legend()
plt.xlabel('n_neighbors')
plt.ylabel('Accuracy')
plt.show()

#After computing the value of neighbors with the highest accuracy
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(flat_face_x_train, face_y_train)
y_predict=knn.predict(flat_face_x_test)
visualize(face_x_test, y_predict, 9, 70, 60)
