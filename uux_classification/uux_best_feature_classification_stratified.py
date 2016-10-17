from uux_classification import uux_data, uux_preprocessing, uux_labelset_stratification

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_selection.univariate_selection import SelectPercentile
from sklearn.feature_selection import chi2
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import metrics
import pylab as pl


#get the data_txt from DB
numDimensions = 22
numFolds = 5

X_train = uux_data.getUUXSentences(numDimensions)
y_train = uux_data.getUUXSentenceDimension(numDimensions)
y_train_binary = MultiLabelBinarizer().fit_transform(y_train)

target_names = uux_data.getUUXDimensions(numDimensions)

x_train_folds, x_test_folds, y_train_folds, y_test_folds = uux_labelset_stratification.kFoldStratify(numFolds)
target_names = uux_data.getUUXDimensions(numDimensions)

percentiles = range(1, 100, 5)
results = []

for perc in range(1, 100, 5):

    p = np.empty([numFolds])
    ch2 = SelectPercentile(chi2, percentile=perc)

    #perfrom 5folds cross-validation
    for i in range(0, numFolds):

        #data_txt preproccessing - tokenization, selecting 90% of the best features
        vectorizer = TfidfVectorizer(tokenizer=uux_preprocessing.tokenize)
        X_train_features = vectorizer.fit_transform(x_train_folds[i])
        X_train_features_names =vectorizer.fit(x_train_folds[i]).vocabulary_

        X_train_features = ch2.fit_transform(X_train_features, y_train_folds[i])
        selected_features_names = np.asarray(vectorizer.get_feature_names())[ch2.get_support()]

        classifier = Pipeline([
            ('tfidf', vectorizer),
            ('chi2', ch2),
            ('clf', OneVsRestClassifier(LinearSVC()))])

        classifier.fit(x_train_folds[i], y_train_folds[i])

        predicted = classifier.predict(x_test_folds[i])

        print metrics.precision_score(y_test_folds[i], predicted)
        p[i] =  metrics.precision_score(y_test_folds[i], predicted)
    print p
    results = np.append(results, p.mean())
    print "Results"
    print results

optimal_precentil = np.where(results == results.max())[0]

print "Optimal percentage: " + str(optimal_precentil) + " optimal number of features: " + str(percentiles[optimal_precentil]);

pl.figure()
pl.xlabel("Number of features selected")
pl.ylabel("Cross-validation precision ")
pl.plot(percentiles, results)
pl.savefig('foo.pdf', bbox_inches='tight')