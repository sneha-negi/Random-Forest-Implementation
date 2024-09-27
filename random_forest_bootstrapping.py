import numpy as np
from sklearn import tree
from bootstrap import draw_bootstrap
from utils.error_score import error_score

def random_forest_bootstrapping(X_train, y_train, n_estimators):
    forest = list()
    oob_ls = list()
    for i in range(n_estimators):
        X_bootstrap, y_bootstrap, X_oob, y_oob = draw_bootstrap(X_train, y_train)
        decision_tree = tree.DecisionTreeClassifier()
        decision_tree.fit(X_bootstrap, y_bootstrap)
        forest.append(decision_tree)
        oob_pred = decision_tree.predict(X_oob)
        oob_error = error_score(oob_pred, y_oob)
        oob_ls.append(oob_error)
    print("OOB error estimate: {:.2f}".format(np.mean(oob_ls)))
    return forest