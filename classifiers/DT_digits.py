import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from starter_code import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

# Reading dataset
x_train = read_lines("../data/digitdata/trainingimages", 28)
y_train = read_labels("../data/digitdata/traininglabels")

x_test = read_lines("../data/digitdata/testimages", 28)
y_test = read_labels("../data/digitdata/testlabels")

# we need to flat our array from 3D to 2D because the model accepts only 2D/1D
flat_x_train = x_train.reshape(5000, 784)
flat_x_test = x_test.reshape(1000, 784)


# Choose the first hyperparameter criterion
def criterion():
    scores = {
        'gini': 0,
        'entropy': 0,
    }
    for criterion in scores.keys():
        model = DecisionTreeClassifier(criterion=criterion)
        model.fit(flat_x_train, y_train)
        scores[criterion] = accuracy_score(model.predict(flat_x_test), y_test)

    # creating the bar plot
    fig = plt.figure()
    plt.bar(scores.keys(), scores.values(),width=0.15,align='edge')
    values=list(scores.values())
    for i in range(len(values)):
        plt.text(i, values[i], values[i])
    plt.xlabel("criterion")
    plt.ylabel("accuracy")
    plt.title("accuracy for each criterion")
    plt.show()

# Choose the second hyperparameter max depth
def max_depth():
    scores = {i:0 for i in range(5,21)}
    depth=range(5,21)
    for maxdepth in depth:
        model = DecisionTreeClassifier(max_depth=maxdepth)
        model.fit(flat_x_train, y_train)
        scores[maxdepth] = accuracy_score(model.predict(flat_x_test), y_test)

        # creating the bar plot

    fig = plt.figure()
    plt.plot(scores.keys(), scores.values())
    plt.xlabel("max depth")
    plt.ylabel("accuracy")
    plt.title("Accuracy for each max depth")
    print(scores)
    plt.show()


def min_samples_split():
    scores = {i:0 for i in range(2,50)}
    samplessplit=range(2,50)
    for minsamplessplit in samplessplit:
        model = DecisionTreeClassifier(min_samples_split=minsamplessplit)
        model.fit(flat_x_train, y_train)
        scores[minsamplessplit] = accuracy_score(model.predict(flat_x_test), y_test)

    fig = plt.figure()
    plt.plot(scores.keys(), scores.values())
    plt.xlabel("min samples split")
    plt.ylabel("accuracy")
    plt.title("Accuracy for each min samples split")
    print(scores)
    plt.show()


def min_samples_leaf():
    scores = {i:0 for i in range(1,50)}
    samplesleaf=range(1,50)
    for minsamplesleaf in samplesleaf:
        model = DecisionTreeClassifier(min_samples_leaf=minsamplesleaf)
        model.fit(flat_x_train, y_train)
        scores[minsamplesleaf] = accuracy_score(model.predict(flat_x_test), y_test)

    fig = plt.figure()
    plt.plot(scores.keys(), scores.values())
    plt.xlabel("min samples leaf")
    plt.ylabel("accuracy")
    plt.title("Accuracy for each min samples leaf")
    print(scores)
    plt.show()

def max_leaf_nodes():
    scores = {i:0 for i in range(2,100)}
    leafnodes=range(2,200)
    for maxleaf in leafnodes:
        model = DecisionTreeClassifier(max_leaf_nodes=maxleaf)
        model.fit(flat_x_train, y_train)
        scores[maxleaf] = accuracy_score(model.predict(flat_x_test), y_test)

    fig = plt.figure()
    plt.plot(scores.keys(), scores.values())
    plt.xlabel("max leaf nodes")
    plt.ylabel("accuracy")
    plt.title("Accuracy for each max leaf nodes")
    print(scores)
    plt.show()



def comination():
    # Create the parameter grid
    param_grid = {
        'max_depth': [5, 10, 15, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_leaf_nodes': [10, 20, 30]
    }

    # Create the decision tree model
    dt = DecisionTreeClassifier()

    # Create the grid search object
    grid_search = GridSearchCV(estimator=dt, param_grid=param_grid, cv=5)

    # Fit the grid search to the data
    grid_search.fit(flat_x_train, y_train)

    # Print the best parameters
    print(grid_search.best_params_)


def main():
    # best
    #criterion=gini
    # max depth=15
    # min samples split=3
    #  min samples leaf=1
    #max leaf nodes =191
   #  {'max_depth': 10, 'max_leaf_nodes': 30, 'min_samples_leaf': 1, 'min_samples_split': 2}

    model = DecisionTreeClassifier(max_depth=10, max_leaf_nodes=30, min_samples_leaf=1, min_samples_split=2)
    model.fit(flat_x_train,y_train)
    print(accuracy_score(y_test,model.predict(flat_x_test)))





if __name__ == '__main__':
    main()
