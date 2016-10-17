__author__ = 'elsabakiu'

from scipy.stats import spearmanr
import numpy as np
from sklearn import metrics


f = open('data_txt/agreements_features_truth_dataset.txt')
truths = f.readlines()
f.close()

f = open('data_txt/agreements_features_aggregated_svm.txt')
svm = f.readlines()
f.close()

f = open('data_txt/agreements_features_aggregated_sentistrength.txt')
sentistr = f.readlines()
f.close()

f = open('data_txt/agreements_features_stanford_sentence.txt')
stanf = f.readlines()
f.close()


truth = np.empty(len(truths))
svms = np.empty(len(svm))
sentistrengths = np.empty(len(sentistr))
stanfs = np.empty(len(stanf))

i = 0;
for t, sent_svm, sent_strength, stan  in zip(truths, svm, sentistr, stanf):
    truth_score = int(t.split("\t")[0])

    svm_score = int(sent_svm.split("\t")[0])
    stan_score = int(stan.split("\t")[0])
    #SCALE SENTISTRENTH PREDICTION
    sentistr_eval = int(sent_strength.split("\t")[0])
    # 3-scale mapping: {2, 3, 4} -> {1}, {-1, 1} -> {0}, {-2, -3, -4} -> {-1}
    scale_3 = 0
    if (sentistr_eval in range(2, 5)):
        scale_3 = 1
    elif (sentistr_eval in range(-1, 2)):
        scale_3 = 0
    elif (sentistr_eval in range(-4, -1)):
        scale_3 = -1

    truth[i], svms[i], sentistrengths[i], stanfs[i] = float(truth_score), float(svm_score), float(scale_3), float(stan_score)
    i = i + 1

print "SENTISTRENGTH: "
print "Precision:"
print metrics.precision_score(truth, sentistrengths)
print "Recall:"
print metrics.recall_score(truth, sentistrengths)

print "Accuracy:"
print metrics.accuracy_score(truth, sentistrengths)

print "F1 Score:"
print metrics.f1_score(truth, sentistrengths)

print "Correlation:"
print spearmanr(truth, sentistrengths)


print "SVM: "
print "Precision:"
print metrics.precision_score(truth, svms)
print "Recall:"
print metrics.recall_score(truth, svms)

print "Accuracy:"
print metrics.accuracy_score(truth, svms)

print "F1 Score:"
print metrics.f1_score(truth, svms)

print "Correlation:"
print spearmanr(truth, svms)

print metrics.confusion_matrix(truth, svms)



print "Stanford"
print "Precision:"
print metrics.precision_score(truth, stanfs)
print "Recall:"
print metrics.recall_score(truth, stanfs)

print "Accuracy:"
print metrics.accuracy_score(truth, stanfs)

print "F1 Score:"
print metrics.f1_score(truth, stanfs)

print "Correlation:"
print spearmanr(truth, stanfs)

print metrics.confusion_matrix(truth, stanfs)
