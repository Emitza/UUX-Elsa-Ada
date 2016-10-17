from sentiment_analysis.machine_learning import negation_handling

__author__ = 'elsabakiu'

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.feature_selection.univariate_selection import SelectPercentile
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2
from sklearn import cross_validation
from sklearn.externals import joblib

f = open('../sentistrength/data_txt/app_store_data/agreements_truth_dataset_3_scale.txt')
lines = f.readlines()
f.close()

sentences = []
sentiments = []

for line in lines:
    row = []
    elements = line.rstrip('\r\n').split('\t')
    sentences.append(elements[1])
    sentiments.append(int(elements[0]))


# SVM Solution
# Data_txt preproccessing - tokenization, selecting the best features
vectorizer = CountVectorizer(tokenizer=negation_handling.tokenize, ngram_range=(1, 1), max_df=0.5, lowercase=False)
tfidfTans = TfidfTransformer(use_idf=True, sublinear_tf=True, smooth_idf=False, norm='l2')

classifier = Pipeline([
    ('vect', vectorizer),
    ('tfidf', tfidfTans),
    ('feature_selection', SelectPercentile(chi2, percentile=93)),
    ('clf', LinearSVC(C=0.10000000000000001, multi_class='ovr')),
])

print "With negation handling: "
skf = cross_validation.StratifiedKFold(sentiments, n_folds=5)
scores = cross_validation.cross_val_score(classifier, sentences, sentiments, cv= skf, scoring='f1')



# print "Without negation handling: "
# vectorizer = CountVectorizer(tokenizer=None, ngram_range=(1, 1), max_df=0.5, lowercase=False)
# tfidfTans = TfidfTransformer(use_idf=True, sublinear_tf=True, smooth_idf=False, norm='l2')
#
# classifier = Pipeline([
#     ('vect', vectorizer),
#     ('tfidf', tfidfTans),
#     ('feature_selection', SelectPercentile(chi2, percentile=93)),
#     ('clf', LinearSVC(C=0.10000000000000001, multi_class='ovr')),
# ])
#
# skf = cross_validation.StratifiedKFold(sentiments, n_folds=5)
# scores = cross_validation.cross_val_score(classifier, sentences, sentiments, cv= skf, scoring='f1')
# print scores
# print sum(scores)/float(len(scores))


# Naive Bayes - Solution
# Data_txt preproccessing - tokenization, selecting the best features

# print "With negation handling:"
#
# vectorizer = CountVectorizer(tokenizer=negation_handling.tokenize, ngram_range=(1, 2), max_df=0.5, lowercase=True)
# tfidfTans = TfidfTransformer(use_idf=True, sublinear_tf=True, smooth_idf=False, norm='l1')
#
# classifier = Pipeline([
#     ('vect', vectorizer),
#     ('tfidf', tfidfTans),
#     ('feature_selection', SelectPercentile(f_classif, percentile=29)),
#     ('clf', MultinomialNB(alpha=0.01)),
# ])
#
# skf = cross_validation.StratifiedKFold(sentiments, n_folds=5)
# scores = cross_validation.cross_val_score(classifier, sentences, sentiments, cv= skf, scoring='f1')
# print scores
# print sum(scores)/float(len(scores))
#
#
# print "Without negation handling:"
#
# vectorizer = CountVectorizer(tokenizer=None, ngram_range=(1, 2), max_df=0.5, lowercase=True)
# tfidfTans = TfidfTransformer(use_idf=True, sublinear_tf=True, smooth_idf=False, norm='l1')
#
# classifier = Pipeline([
#     ('vect', vectorizer),
#     ('tfidf', tfidfTans),
#     ('feature_selection', SelectPercentile(f_classif, percentile=29)),
#     ('clf', MultinomialNB(alpha=0.01)),
# ])
# skf = cross_validation.StratifiedKFold(sentiments, n_folds=5)
# scores = cross_validation.cross_val_score(classifier, sentences, sentiments, cv= skf, scoring='f1')

classifier.fit(sentences, sentiments)
joblib.dump(classifier, 'classifier/sentiment_classifier.pkl')
