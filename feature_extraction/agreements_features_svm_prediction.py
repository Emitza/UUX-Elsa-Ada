__author__ = 'elsabakiu'

from sklearn.externals import joblib
import numpy as np

f = open('data_txt/agreements_features_unlabeled_dataset.txt')
lines = f.readlines()
f.close()

f = open('data_txt/agreements_features_svm.txt', 'r+')

clf = joblib.load('../sentiment_analysis/machine_learning/classifier/sentiment_classifier.pkl')
print clf

for line in lines:
    line = line.split("\t")
    sentence = line[0].rstrip('\r\n')
    sentiment = clf.predict([sentence])
    project_id = str(line[1])
    f.write(str(sentiment[0]) + "\t" + sentence + "\t" + project_id)

f.close();



