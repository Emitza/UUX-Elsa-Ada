from sentiment_analysis.machine_learning import negation_handling

__author__ = 'elsabakiu'

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection.univariate_selection import SelectPercentile
from sklearn.feature_selection import chi2
from sklearn import cross_validation
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


f = open('../sentistrength/data_txt/combined/truth_dataset_3_scale.txt')
lines = f.readlines()
f.close()

sentences = []
sentiments = []

for line in lines:
    row = []
    elements = line.rstrip('\r\n').split('\t')
    sentences.append(elements[1])
    sentiments.append(int(elements[0]))

ch2 = SelectPercentile(chi2, percentile=96)

# define a pipeline combining a text feature extractor with a simple
# classifier
pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB()),
])

parameters = {
    'tfidf__use_idf': (True, False)
}

# K-Fold cross-validation strategy
skf = cross_validation.StratifiedKFold(sentences, n_folds=5)

mnb_grid = GridSearchCV(pipeline, parameters, scoring='precision', cv=skf, n_jobs=4)
mnb_grid.fit(sentences, sentiments)
print(mnb_grid.best_params_)
print (mnb_grid.best_score_)