import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_selection.univariate_selection import SelectPercentile
from sklearn.feature_selection import chi2
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from uux_classification import uux_data, uux_preprocessing
from sklearn.externals import joblib


#get the data_txt from DB
numDimensions = 22
numFolds = 5

X_train = uux_data.getUUXSentences(numDimensions)
y_train = uux_data.getUUXSentenceDimension(numDimensions)
y_train_binary = MultiLabelBinarizer().fit_transform(y_train)

target_names = uux_data.getUUXDimensions(numDimensions)


#data_txt preproccessing - tokenization, selecting 90% of the best features
vectorizer = TfidfVectorizer(tokenizer=uux_preprocessing.tokenize)
X_train_features = vectorizer.fit_transform(X_train)
X_train_features_names = vectorizer.fit(X_train).vocabulary_

ch2 = SelectPercentile(chi2, percentile=16)
X_train_features = ch2.fit_transform(X_train_features, y_train_binary)
selected_features_names = np.asarray(vectorizer.get_feature_names())[ch2.get_support()]
print str(len(selected_features_names))

classifier = Pipeline([
    ('tfidf', vectorizer),
    ('chi2', ch2),
    ('clf', OneVsRestClassifier(LinearSVC()))])

classifier.fit(X_train, y_train_binary)
joblib.dump(classifier, 'classifier/uux_classifier.pkl')