import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_selection.univariate_selection import SelectPercentile
from sklearn.feature_selection import chi2
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report
from sklearn.metrics import metrics
from uux_classification import uux_data, uux_preprocessing
from uux_classifier_evaluation import UUX_classifier_evaluation


#get the data_txt from DB
numDimensions = 22
numFolds = 5

X_train = uux_data.getUUXSentences(numDimensions)
y_train = uux_data.getUUXSentenceDimension(numDimensions)
y_train_binary = MultiLabelBinarizer().fit_transform(y_train)

target_names = uux_data.getUUXDimensions(numDimensions)

# folds_X_train = np.split(X_train, [3732])
# folds_y_train = np.split(y_train_binary, [3732])

folds_X_train = np.split(X_train, [933, 1866, 2799, 3732])
folds_y_train = np.split(y_train_binary, [933, 1866, 2799, 3732])

p = np.empty([numFolds, numDimensions])
r = np.empty([numFolds, numDimensions])
f1 = np.empty([numFolds, numDimensions])
s = np.empty([numFolds, numDimensions])

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

    ch2 = SelectPercentile(chi2, percentile=16)
    X_train_features = ch2.fit_transform(X_train_features, fold_y_train)
    selected_features_names = np.asarray(vectorizer.get_feature_names())[ch2.get_support()]
    print str(len(selected_features_names))

    classifier = Pipeline([
        ('tfidf', vectorizer),
        ('chi2', ch2),
        ('clf', OneVsRestClassifier(LinearSVC()))])

    classifier.fit(fold_x_train, fold_y_train)

    predicted = classifier.predict(fold_x_test)

    print classification_report(fold_y_test, predicted, target_names=target_names)
    p[i], r[i], f1[i], s[i] = metrics.precision_recall_fscore_support(fold_y_test, predicted)

    eval = UUX_classifier_evaluation(fold_y_train, fold_x_test, fold_y_test, numDimensions)
    pValues = eval.calculate_significance(predicted)

    for label, pValue in zip(target_names, pValues):
        print label + "&" + str(pValue) + "\\\\"

#Aggregate the predicted results(avg) and print them
for i in range (0, numDimensions):
    label = target_names[i]
    pLabel, rLabel, f1Label= 0, 0, 0;

    for j in range (0, numFolds):
        pLabel = pLabel + p[j][i]
        rLabel = rLabel + r[j][i]
        f1Label = f1Label + f1[j][i]

    print label + "&" + str(round(pLabel/numFolds, 2)) + "&" + str(round(rLabel/numFolds, 2)) + "&" + str(round(f1Label/numFolds, 2))+"\\\\";
    # print 'Precision  ' + str(pLabel/numFolds)
    # print 'Recall  ' + str(rLabel/numFolds)
    # print 'F1 measure  ' + str(f1Label/numFolds)
    # print '-------------------------'
