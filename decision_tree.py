from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt
import numpy as np


def decision_tree(X, Y):
    Y = list(map(lambda x: round(x),Y))
    Y = np.array(Y, dtype=int)
    clf = tree.DecisionTreeClassifier(max_depth=10, max_features='auto')
    clf = clf.fit(X, Y)
    tree.plot_tree(clf)
    plt.show()
    print(cross_val_score(clf, X, Y))
    print(classification_report(Y, clf.predict(X)))
    