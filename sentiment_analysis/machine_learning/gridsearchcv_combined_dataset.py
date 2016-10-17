from __future__ import print_function

from pprint import pprint
from time import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection.univariate_selection import SelectPercentile
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import chi2
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sentiment_analysis.machine_learning import negation_handling
import numpy as np

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

print("%d documents" % len(sentences))
print("%d categories" % len(sentiments))
print()

###############################################################################
# define a pipeline combining a text feature extractor with a simple
# classifier
pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('feature_selection', SelectPercentile()),
    ('clf', LinearSVC()),
])

# uncommenting more parameters will give better exploring power but will
# increase processing time in a combinatorial way
parameters = {
    'vect__max_df': (0.5, 0.75, 1.0),
    'vect__ngram_range': ((1, 1), (1, 2), (2, 2)),  # unigrams or bigrams
    'vect__tokenizer': (negation_handling.tokenize, None),
    'vect__lowercase': (True , False),
    'tfidf__use_idf': (True, False),
    'tfidf__smooth_idf': (True, False),
    'tfidf__sublinear_tf': (True, False),
    'tfidf__norm': ('l1', 'l2'),
    'clf__multi_class': ('ovr', 'crammer_singer'),
    'clf__C': np.logspace(-1, 1, 3),
    'feature_selection__score_func': (chi2, f_classif),
    'feature_selection__percentile': (25, 90)
}

if __name__ == "__main__":
    # multiprocessing requires the fork to happen in a __main__ protected
    # block

    # find the best parameters for both the feature extraction and the
    # classifier
    grid_search = GridSearchCV(pipeline, parameters, scoring='precision', n_jobs=-1, verbose=1)

    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    pprint(parameters)
    t0 = time()

    grid_search.fit(sentences, sentiments)
    print("done in %0.3fs" % (time() - t0))
    print()

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))