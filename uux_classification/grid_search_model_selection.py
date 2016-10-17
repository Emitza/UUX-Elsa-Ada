from __future__ import print_function

from pprint import pprint
from time import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import f_classif
from sklearn.grid_search import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.feature_selection.univariate_selection import SelectPercentile
from sklearn.feature_selection import chi2
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sentiment_analysis.machine_learning import negation_handling
from uux_classification import uux_data
from sklearn.cross_validation import KFold
from sklearn.pipeline import Pipeline
import numpy as np

X_train = uux_data.getUUXSentences(22)
y_train = uux_data.getUUXSentenceDimension(22)
y_train_binary = MultiLabelBinarizer().fit_transform(y_train)
target_names = uux_data.getUUXDimensions(22)



###############################################################################
# define a pipeline combining a text feature extractor with a simple
# classifier
pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('feature_selection', SelectPercentile()),
    ('clf', OneVsRestClassifier(LinearSVC())),
])

# uncommenting more parameters will give better exploring power but will
# increase processing time in a combinatorial way
parameters = {
    'vect__max_df': (0.5, 0.75, 1.0),
    'vect__ngram_range': ((1, 1), (1, 2), (2, 2)),  # unigrams or bigrams
    'vect__lowercase': (True , False),
    'tfidf__use_idf': (True, False),
    'tfidf__smooth_idf': (True, False),
    'tfidf__sublinear_tf': (True, False),
    'tfidf__norm': ('l1', 'l2'),
    'feature_selection__score_func': (chi2, f_classif),
    'feature_selection__percentile': (25, 90)
}

if __name__ == "__main__":
    # multiprocessing requires the fork to happen in a __main__ protected
    # block

    # find the best parameters for both the feature extraction and the
    # classifier
    grid_search = GridSearchCV(pipeline, parameters, cv = KFold(n=len(X_train), n_folds=5), scoring='precision', n_jobs=-1, verbose=1)

    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    pprint(parameters)
    t0 = time()

    grid_search.fit(X_train, np.array(y_train_binary))
    print("done in %0.3fs" % (time() - t0))
    print()

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))



