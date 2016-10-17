from uux_classification import uux_data

__author__ = 'elsabakiu'

from sklearn import cross_validation
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np

def kFoldStratify(nFolds):
    numDimensions = 22
    sentence_ids, labels = uux_data.getBinaryLabels(numDimensions)
    sentences = uux_data.getUUXSentences(numDimensions)

    y = uux_data.getUUXSentenceDimension(numDimensions)
    y_binary = MultiLabelBinarizer().fit_transform(y)

    X_train_folds = []
    X_test_folds = []
    Y_train_folds = []
    Y_test_folds = []

    folds = cross_validation.StratifiedKFold(labels, n_folds=nFolds)


    for train_index, test_index in folds:
        X_train, X_test = sentences[train_index], sentences[test_index]
        y_train, y_test = y_binary[train_index], y_binary[test_index]
        X_train_folds.append(np.array(X_train))
        X_test_folds.append(np.array(X_test))
        Y_train_folds.append(np.array(y_train))
        Y_test_folds.append(np.array(y_test))

    return X_train_folds, X_test_folds, Y_train_folds, Y_test_folds