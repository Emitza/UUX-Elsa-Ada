from uux_classification import uux_data, uux_preprocessing

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


folds_X_train = np.split(X_train, [933, 1866, 2799, 3732])
folds_y_train = np.split(y_train_binary, [933, 1866, 2799, 3732])

percentiles = range(1, 100, 5)
results = []

for perc in range(1, 100, 5):

    p = np.empty([numFolds])
    ch2 = SelectPercentile(chi2, percentile=perc)

    #perfrom 5folds cross-validation
    for i in range(0, numFolds):

        fold_x_train = np.array([], dtype=np.double)
        fold_y_train = np.array([], dtype=np.double).reshape(0, numDimensions)

        #Conctatenate the 4 folds for training
        for j in range(0, numFolds):
            if(i != j):
                fold_x_train = np.concatenate((fold_x_train, folds_X_train[j]))
                fold_y_train = np.concatenate((fold_y_train, folds_y_train[j]))

        fold_x_test = folds_X_train[i]
        fold_y_test = folds_y_train[i]

        #data_txt preproccessing - tokenization, selecting 90% of the best features
        vectorizer = TfidfVectorizer(tokenizer=uux_preprocessing.tokenize)
        X_train_features = vectorizer.fit_transform(fold_x_train)
        X_train_features_names =vectorizer.fit(fold_x_train).vocabulary_

        X_train_features = ch2.fit_transform(X_train_features, fold_y_train)
        selected_features_names = np.asarray(vectorizer.get_feature_names())[ch2.get_support()]

        classifier = Pipeline([
            ('tfidf', vectorizer),
            ('chi2', ch2),
            ('clf', OneVsRestClassifier(LinearSVC()))])

        classifier.fit(fold_x_train, fold_y_train)

        predicted = classifier.predict(fold_x_test)

        print metrics.precision_score(fold_y_test, predicted)
        p[i] =  metrics.precision_score(fold_y_test, predicted)
    print p
    results = np.append(results, p.mean())
    print "Results"
    print results

optimal_precentil = np.where(results == results.max())[0]

print "Optimal percentage: " + str(optimal_precentil) + " optimal number of features: " + str(percentiles[optimal_precentil]);

pl.figure()
pl.xlabel("Percentage of selected features")
pl.ylabel("Cross-validation precision ")
pl.plot(percentiles, results)
pl.savefig('precision_score.pdf', bbox_inches='tight')
