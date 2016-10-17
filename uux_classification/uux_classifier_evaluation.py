__author__ = 'elsabakiu'

import numpy as np
from scikits.statsmodels.sandbox.stats import runs

class UUX_classifier_evaluation:

    def __init__(self, label_train_data, example_test_data, label_test_data, num_dimensions):
        self.label_train_data = label_train_data
        self.example_test_data = example_test_data
        self.label_test_data = label_test_data
        self.num_dimensions = num_dimensions

    #Define the prediction of the baseline classifier as the majority class of each label
    def base_classifier(self):
        base_classifier_prediction = []

        majority_class = []
        for i in range(0, self.num_dimensions):
                majority_class.append(self.majority_class(i))

        for i in range(0, len(self.example_test_data)):
            row_prediction = []
            for j in range(0, self.num_dimensions):
                row_prediction.append(self.majority_class(j))
            base_classifier_prediction.append(row_prediction)

        return np.asarray(base_classifier_prediction)


    def correctness_predictions(self, prediction):
        prediction = np.asarray(prediction).T
        actual = np.asarray(self.label_test_data).T

        correctness_matrix = []
        for i in range(0, self.num_dimensions):
            correctness_row = []
            for j in range(0, len(self.example_test_data)):
                correctness_row.append(1 if prediction[i][j] == actual[i][j] else 0)
            correctness_matrix.append(correctness_row)

        return np.asarray(correctness_matrix)


    def calculate_significance(self, classifier_prediction):
        classifier_correctness_matrix = self.correctness_predictions(classifier_prediction)
        baseline_correctness_matrix = self.correctness_predictions(self.base_classifier())

        stats = []
        pvalue = []

        for i in range(0, self.num_dimensions):
            statVal, pvalueVal = runs.mcnemar(classifier_correctness_matrix[i], baseline_correctness_matrix[i], False)
            stats.append(statVal)
            pvalue.append(pvalueVal)
        return pvalue


    def majority_class(self, label):
        num_0_examples = 0
        num_1_examples = 0

        for i in range(0, len(self.label_test_data)):

            if (self.label_test_data[i][label] == 0):
                num_0_examples += 1
            else:
                num_1_examples += 1

        return 0 if num_0_examples >= num_1_examples else num_1_examples


