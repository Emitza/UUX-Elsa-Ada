__author__ = 'elsabakiu'

import MySQLdb
import arff
import math

db = MySQLdb.connect('localhost', 'root', 'root', 'feature_extraction')
# prepare a cursor object using cursor() method
cursor = db.cursor()

# execute SQL query using execute() method.
sentences_sql = "SELECT a.sentence, a.score1 s1, a.score2 s2, a.label1, r.project_id FROM `agreements` a INNER JOIN reviews r on a.review_id = r.review_id ORDER BY sentence"
try:
   # Execute the SQL command
   cursor.execute(sentences_sql)
   sentences = cursor.fetchall()

   f = open('data_txt/agreements_features_truth_dataset.txt', 'r+')

   for sentence in sentences:
      sent = sentence[0].replace("\n", "")
      sent = sent.replace("\t", " ")
      if (sentence[1] == None):
          score_avg = int(sentence[2])
      elif (sentence[2] == None):
          score_avg = int(sentence[1])
      else:
          score_avg = int((int(sentence[1]) + int(sentence[2])) / 2)

      truth_3_scale = 0
      if (score_avg > 0):
        truth_3_scale = 1
      elif (score_avg == 0):
        truth_3_scale = 0
      elif (score_avg < 0):
        truth_3_scale = -1

      f.write(str(truth_3_scale) + "\t" + sent + "\t" + sentence[3] + "\t" + str(sentence[4]) + "\n")

   f.close()

except Exception, e:
   print e

# disconnect from server
db.close()