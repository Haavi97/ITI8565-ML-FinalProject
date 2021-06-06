from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, f1_score, accuracy_score
from matplotlib import pyplot as plt
import numpy as np


def decision_tree(X, Y, depth=10, do_plot=True):
    Y = list(map(lambda x: round(x),Y))
    Y = np.array(Y, dtype=int)
    clf = tree.DecisionTreeClassifier(max_depth=depth, max_features='auto')
    clf = clf.fit(X, Y)
    if do_plot:
        tree.plot_tree(clf)
        plt.show()
    print(cross_val_score(clf, X, Y))
    y_predicted = clf.predict(X)
    print('Decision tree of max depth {} classification report:'.format(depth))
    print(classification_report(Y, y_predicted))
    return f1_score(Y, y_predicted), accuracy_score(Y, y_predicted)

def tree_depths(X, Y):
    f1s = []
    accus = []
    for e in range(3,20):
        f1_, accu_ = decision_tree(X, Y, depth=e, do_plot=False)
        f1s.append(f1_)
        accus.append(accu_)
    x = range(3,20)
    plt.plot(x, f1s, label='F1 score')
    plt.plot(x, accus, label='Accuracy')
    plt.xlabel('Maximum depth')
    plt.legend()
    plt.show()
    