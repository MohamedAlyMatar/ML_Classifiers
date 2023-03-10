import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sources.starter_code import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import export_graphviz
import graphviz
from sklearn.ensemble import RandomForestClassifier
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
        model = DecisionTreeClassifier(criterion=criterion,random_state=0)
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
    max_key = max(scores, key=scores.get)
    print(f"The criterion which maximizes the accuracy is: {max_key}")
    plt.show()
    return max_key

# Choose the second hyperparameter max depth
def max_depth():
    scores = {i:0 for i in range(2,21)}
    depth=range(2,21)
    for maxdepth in depth:
        model = DecisionTreeClassifier(max_depth=maxdepth,random_state=0)
        model.fit(flat_x_train, y_train)
        scores[maxdepth] = accuracy_score(model.predict(flat_x_test), y_test)

    # creating the bar plot
    fig = plt.figure()
    plt.plot(scores.keys(), scores.values())
    plt.xlabel("max depth")
    plt.ylabel("accuracy")
    plt.title("Accuracy for each max depth")
    max_key = max(scores, key=scores.get)
    print(f"The max depth value which maximizes the accuracy is: {max_key}")
    plt.show()
    return max_key


def min_samples_split():
    scores = {i:0 for i in range(2,50)}
    samplessplit=range(2,50)
    for minsamplessplit in samplessplit:
        model = DecisionTreeClassifier(min_samples_split=minsamplessplit,random_state=0)
        model.fit(flat_x_train, y_train)
        scores[minsamplessplit] = accuracy_score(model.predict(flat_x_test), y_test)

    fig = plt.figure()
    plt.plot(scores.keys(), scores.values())
    plt.xlabel("min samples split")
    plt.ylabel("accuracy")
    plt.title("Accuracy for each min samples split")
    max_key = max(scores, key=scores.get)
    print(f"The min samples split value which maximizes the accuracy is: {max_key}")
    plt.show()
    return max_key


def min_samples_leaf():
    scores = {i:0 for i in range(1,50)}
    samplesleaf=range(1,50)
    for minsamplesleaf in samplesleaf:
        model = DecisionTreeClassifier(min_samples_leaf=minsamplesleaf,random_state=0)
        model.fit(flat_x_train, y_train)
        scores[minsamplesleaf] = accuracy_score(model.predict(flat_x_test), y_test)

    fig = plt.figure()
    plt.plot(scores.keys(), scores.values())
    plt.xlabel("min samples leaf")
    plt.ylabel("accuracy")
    plt.title("Accuracy for each min samples leaf")
    max_key = max(scores, key=scores.get)
    print(f"The min samples leaf value which maximizes the accuracy is: {max_key}")
    plt.show()
    return max_key

def max_leaf_nodes():
    scores = {i:0 for i in range(2,100)}
    leafnodes=range(2,200)
    for maxleaf in leafnodes:
        model = DecisionTreeClassifier(max_leaf_nodes=maxleaf,random_state=0)
        model.fit(flat_x_train, y_train)
        scores[maxleaf] = accuracy_score(model.predict(flat_x_test), y_test)

    fig = plt.figure()
    plt.plot(scores.keys(), scores.values())
    plt.xlabel("max leaf nodes")
    plt.ylabel("accuracy")
    plt.title("Accuracy for each max leaf nodes")
    max_key = max(scores, key=scores.get)
    print(f"The max leaf nodes value which maximizes the accuracy is: {max_key}")
    plt.show()
    return max_key



def combination():
    model = DecisionTreeClassifier(min_samples_split=4, criterion='gini', random_state=0, min_samples_leaf=1)
    model.fit(flat_x_train, y_train)
    print("accuracy:", accuracy_score(model.predict(flat_x_test), y_test))
    return model


def visualize_tree():
    model=combination()
    dot_data = export_graphviz(model, filled=True,out_file=None)
    graph = graphviz.Source(dot_data,format="pdf")
    graph
    graph.render("decision_tree_graphivz")

def randomforest():

    model=RandomForestClassifier(n_estimators=200,random_state=0)
    model.fit(flat_x_train,y_train)
    print("accuracy:", accuracy_score(model.predict(flat_x_test), y_test))






def main():
    model=combination()
    visualize(x_test, model.predict(flat_x_test), 9, 28, 28)


if __name__ == '__main__':
    main()
