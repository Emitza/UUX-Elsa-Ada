__author__ = 'elsabakiu'

from scipy.stats import spearmanr
import numpy as np
from sklearn import metrics

# f = open('sentistrength/data_txt/uux_data/uux_truth_dataset.txt')
# truths = f.readlines()
# f.close()
# f = open('sentistrength/data_txt/uux_data/uux_stanford_dataset.txt')
# stanfords = f.readlines()
# f.close()
# f = open('sentistrength/data_txt/uux_data/uux_data_optimization_scale.txt')
# sentistrengths = f.readlines()
# f.close()

f = open('sentistrength/data_txt/app_store_data/agreements_truth_dataset_3_scale.txt')
truths = f.readlines()
f.close()
# f = open('sentistrength/data_txt/uux_data/agreements_stanford.txt')
# stanfords = f.readlines()
# f.close()
f = open('sentistrength/data_txt/app_store_data/agreements_optimization_scale.txt')
sentistrengths = f.readlines()
f.close()

f = open('sentistrength/data_txt/app_store_data/agreements_features_svm.txt')
svms = f.readlines()
f.close()


truth = np.empty(len(truths))
#stanford = np.empty(len(stanfords))
sentistrength = np.empty(len(sentistrengths))
svm = np.empty(len(svms))
combined = np.empty(len(truths))

i = 0;
for t, sent, svm_sent in zip(truths, sentistrengths, svms):
    truth_score = int(t.split("\t")[0])
    svm_score = int(svm_sent.split("\t")[0])
    # 3-scale mapping: {1,2} -> {1}, {0} -> {0}, {-1, -2} -> {-1}
    truth_3_scale = 0
    if (truth_score > 0):
        truth_3_scale = 1
    elif (truth_score == 0):
        truth_3_scale = 0
    elif (truth_score < 0):
        truth_3_scale = -1


    senti_parts = sent.split("\t")

    #BINARY SENTISTRENTH PREDICTION
    # positive_sentiment = int(senti_parts[0])
    # negative_sentiment = int(senti_parts[1])
    # sentistr_eval = negative_sentiment if max(positive_sentiment, abs(negative_sentiment)) == abs(negative_sentiment) else positive_sentiment
    # scale_3 = 0
    # if (sentistr_eval in range(2, 6)):
    #     scale_3 = 1
    # elif (sentistr_eval in range(-1, 2)):
    #     scale_3 = 0
    # elif (sentistr_eval in range(-5, -1)):
    #     scale_3 = -1


    #SCALE SENTISTRENTH PREDICTION
    sentistr_eval = int(senti_parts[0])
    # 3-scale mapping: {2, 3, 4} -> {1}, {-1, 1} -> {0}, {-2, -3, -4} -> {-1}
    scale_3 = 0
    if (sentistr_eval in range(2, 5)):
        scale_3 = 1
    elif (sentistr_eval in range(-1, 2)):
        scale_3 = 0
    elif (sentistr_eval in range(-4, -1)):
        scale_3 = -1


    truth[i], sentistrength[i], svm[i] = float(truth_3_scale), float(scale_3), float(svm_score)
    i = i + 1

print "SENTISTRENGTH"
print "Precision:"
print metrics.precision_score(truth, sentistrength)
print "Recall:"
print metrics.recall_score(truth, sentistrength)

print "Accuracy:"
print metrics.accuracy_score(truth, sentistrength)

print "F1 Score:"
print metrics.f1_score(truth, sentistrength)

print "Correlation:"
print spearmanr(truth, sentistrength)

print metrics.confusion_matrix(truth, sentistrength)



print ""
print ""
print "SVM"
print "Precision:"
print metrics.precision_score(truth, svm)
print "Recall:"
print metrics.recall_score(truth, svm)

print "Accuracy:"
print metrics.accuracy_score(truth, svm)

print "F1 Score:"
print metrics.f1_score(truth, svm)

print "Correlation:"
print spearmanr(truth, svm)


print metrics.confusion_matrix(truth, svm)


# print "Precision:"
# print metrics.precision_score(truth, stanford)
# print "Accuracy:"
# print metrics.accuracy_score(truth, stanford)
#
# print metrics.precision_score(truth, combined)
# print metrics.accuracy_score(truth, combined)
# print spearmanr(truth, stanford)
# print spearmanr(truth, combined)