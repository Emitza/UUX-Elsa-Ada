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

f = open('sentistrength/data_txt/app_store_data/agreements_truth_dataset.txt')
truths = f.readlines()
f.close()
f = open('sentistrength/data_txt/app_store_data/agreements_stanford.txt')
stanfords = f.readlines()
f.close()
f = open('sentistrength/data_txt/app_store_data/agreements_optimization_binary.txt')
sentistrengths = f.readlines()
f.close()

truth = np.empty(len(truths))
stanford = np.empty(len(stanfords))
sentistrength = np.empty(len(sentistrengths))
combined = np.empty(len(truths))

i = 0;
for t, stan, sent in zip(truths, stanfords, sentistrengths):
    senti_parts = sent.split("\t")


    #BINARY SENTISTRENTH PREDICTION
    positive_sentiment = int(senti_parts[0])
    negative_sentiment = int(senti_parts[1])

    sentistr_eval = negative_sentiment if max(positive_sentiment, abs(negative_sentiment)) == abs(negative_sentiment) else positive_sentiment

    #Equalty -> considering positive
    #sentistr_eval = positive_sentiment if max(positive_sentiment, abs(negative_sentiment)) == positive_sentiment else negative_sentiment

    scale_5 = 0
    if (sentistr_eval in range(3, 6)):
        scale_5 = 2
    elif (sentistr_eval in range(2, 3)):
        scale_5 = 1
    elif (sentistr_eval in range(-1, 2)):
        scale_5 = 0
    elif (sentistr_eval in range(-2, -1)):
        scale_5 = -1
    elif (sentistr_eval in range(-5, -2)):
        scale_5 = -2

    #SCALE SENTISTRENTH PREDICTION
    sentistr_eval = int(senti_parts[0])
    scale_5 = 0
    if (sentistr_eval in range(3, 5)):
        scale_5 = 2
    elif (sentistr_eval in range(-2, 3)):
        scale_5 = sentistr_eval
    elif (sentistr_eval in range(-4, -2)):
        scale_5 = -2


    truth[i], stanford[i], sentistrength[i] = float(t.split("\t")[0]), float(stan.split("\t")[0]), float(scale_5)
    if(sentistrength[i] == 0):
        combined[i] = 0
    else:
        combined[i] = stanford[i]
    i = i + 1


print "Precision:"
print metrics.precision_score(truth, sentistrength)
print "Accuracy:"
print metrics.accuracy_score(truth, sentistrength)
print "Correlation:"
print spearmanr(truth, sentistrength)


# print "Precision:"
# print metrics.precision_score(truth, stanford)
# print "Accuracy:"
# print metrics.accuracy_score(truth, stanford)
# print metrics.precision_score(truth, combined)
# print metrics.accuracy_score(truth, combined)
# print spearmanr(truth, stanford)
# print spearmanr(truth, combined)