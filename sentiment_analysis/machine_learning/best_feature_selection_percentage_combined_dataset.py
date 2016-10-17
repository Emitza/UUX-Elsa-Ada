from sentiment_analysis.machine_learning import negation_handling

__author__ = 'elsabakiu'

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection.univariate_selection import SelectPercentile
from sklearn.feature_selection import chi2
from sklearn import cross_validation
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import metrics
import pylab as pl
from sklearn.svm import LinearSVC

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import f_classif
from sklearn.feature_extraction.text import TfidfTransformer

f = open('../sentistrength/data_txt/app_store_data/agreements_truth_dataset_3_scale.txt')
lines = f.readlines()
f.close()

sentences = []
sentiments = []


percentiles = range(1, 100, 2)
results = []

for line in lines:
    row = []
    elements = line.rstrip('\r\n').split('\t')
    sentences.append(elements[1])
    sentiments.append(int(elements[0]))


# Support Vector Machine
for perc in range(1, 100, 2):

    vectorizer = CountVectorizer(tokenizer=negation_handling.tokenize, ngram_range=(1, 1), max_df=0.75, lowercase=True)
    tfidfTans = TfidfTransformer(use_idf=True, sublinear_tf=False, smooth_idf=False, norm='l1')

    classifier = Pipeline([
        ('vect', vectorizer),
        ('tfidf', tfidfTans),
        ('feature_selection', SelectPercentile(chi2, percentile=perc)),
        ('clf', LinearSVC(C=0.10000000000000001, multi_class='ovr')),
    ])

    scores = cross_validation.cross_val_score(classifier, sentences, sentiments, cv= 5, scoring='precision')
    results = np.append(results, scores.mean())
    skf = cross_validation.StratifiedKFold(sentiments, n_folds=10)


# Multinomial Naive Bayes
# for perc in range(1, 100, 2):
#
#     vectorizer = CountVectorizer(tokenizer=negation_handling.tokenize, ngram_range=(1, 2), max_df=0.5, lowercase=True)
#     tfidfTans = TfidfTransformer(use_idf=True, sublinear_tf=True, smooth_idf=False, norm='l1')
#
#     classifier = Pipeline([
#         ('vect', vectorizer),
#         ('tfidf', tfidfTans),
#         ('feature_selection', SelectPercentile(f_classif, percentile=perc)),
#         ('clf', MultinomialNB(alpha=0.01)),
#     ])
#
#     scores = cross_validation.cross_val_score(classifier, sentences, sentiments, cv= 5, scoring='precision')
#     results = np.append(results, scores.mean())
#     skf = cross_validation.StratifiedKFold(sentiments, n_folds=10)


optimal_precentil = np.where(results == results.max())[0]


print "Optimal percentage: " + str(optimal_precentil) + " optimal number of features: " + str(percentiles[optimal_precentil]);

pl.figure()
pl.xlabel("Percentage of selected features")
pl.ylabel("Cross-validation precision ")
pl.plot(percentiles, results)
pl.savefig('precision_score_svm.pdf', bbox_inches='tight')