import numpy as np
from sklearn import tree
from bagging import draw_bagging_indices
from utils.error_score import error_score

def random_forest_bagging(X_train, y_train, n_estimators):
    forest = list()
    bagging_indices = draw_bagging_indices(X_train.shape[1], n_estimators)
    for i in range(n_estimators):
        X_bag, y_bag = X_train[:, bagging_indices[i]], y_train
        decision_tree = tree.DecisionTreeClassifier()
        decision_tree.fit(X_bag, y_bag)
        forest.append(decision_tree)
    return forest, bagging_indices