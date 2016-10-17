from uux_classification import uux_data, uux_preprocessing, uux_labelset_stratification
from uux_classifier_evaluation import UUX_classifier_evaluation
__author__ = 'elsabakiu'

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_selection.univariate_selection import SelectPercentile
from sklearn.feature_selection import chi2
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import metrics

numFolds, numDimensions = 5, 22
x_train_folds, x_test_folds, y_train_folds, y_test_folds = uux_labelset_stratification.kFoldStratify(numFolds)
target_names = uux_data.getUUXDimensions(numDimensions)

p = np.empty([numFolds, numDimensions])
r = np.empty([numFolds, numDimensions])
f1 = np.empty([numFolds, numDimensions])
s = np.empty([numFolds, numDimensions])

#perfrom 5folds cross-validation
for i in range(0, numFolds):

    #data_txt preproccessing - tokenization, selecting 90% of the best features
    vectorizer = TfidfVectorizer(tokenizer=uux_preprocessing.tokenize)
    X_train_features = vectorizer.fit_transform(x_train_folds[i])
    X_train_features_names =vectorizer.fit(x_train_folds[i]).vocabulary_

    ch2 = SelectPercentile(chi2, percentile=16)
    X_train_features = ch2.fit_transform(X_train_features, y_train_folds[i])
    selected_features_names = np.asarray(vectorizer.get_feature_names())[ch2.get_support()]

    classifier = Pipeline([
        ('tfidf', vectorizer),
        ('chi2', ch2),
        ('clf', OneVsRestClassifier(LinearSVC()))])

    classifier.fit(x_train_folds[i], y_train_folds[i])

    predicted = classifier.predict(x_test_folds[i])
    print classification_report(y_test_folds[i], predicted, target_names=target_names)
    p[i], r[i], f1[i], s[i] = metrics.precision_recall_fscore_support(y_test_folds[i], predicted)


#Aggregate the predictied results(avg) and print them
for i in range (0, numDimensions):
    label = target_names[i]
    pLabel, rLabel, f1Label= 0, 0, 0;

    for j in range (0, numFolds):
        pLabel = pLabel + p[j][i]
        rLabel = rLabel + r[j][i]
        f1Label = f1Label + f1[j][i]

    print label
    print 'Precision  ' + str(pLabel/numFolds)
    print 'Recall  ' + str(rLabel/numFolds)
    print 'F1 measure  ' + str(f1Label/numFolds)
    print '-------------------------'



# eval = UUX_classifier_evaluation(y_train_folds[0], x_test_folds[0], y_test_folds[0], numDimensions)
# pValues = eval.calculate_significance(predicted)
#
# for label, pValue in zip(target_names, pValues):
#     print label + "  " + str(pValue)

