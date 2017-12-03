from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np
import matplotlib.pyplot as plt


def fit_roc(x, y, tree, backend=None):
    pred = tree.evaluate(*x.T, backend=backend)[0]
    if np.isnan(pred).any():
        return float('-inf')
    return roc_auc_score(y, pred)


def accuracy_from_roc(y_true, fpr, tpr, thresholds):
    y_true = y_true == 1
    pos = y_true.sum()
    neg = (1 - y_true).sum()
    tp = tpr * pos
    tn = (1 - fpr) * neg
    return (tp + tn) / (pos + neg)


def show_roc_curve(title, y_true, pred):
    fpr, tpr, thresholds = roc_curve(y_true, pred)
    plt.title(title)
    plt.plot(fpr, tpr, color='darkorange',
             label='ROC curve (area = %0.5f)' % roc_auc_score(y_true, pred))
    plt.legend(loc="lower right")
    plt.show()


def optimum_threshold(y_true, scores):
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    acc = accuracy_from_roc(y_true, fpr, tpr, thresholds)
    return thresholds[acc.argmax()]


def accuracy(Y1, Y2):
    return 1 - abs((np.array(Y1) == 1) ^ (np.array(Y2) == 1)).mean()
