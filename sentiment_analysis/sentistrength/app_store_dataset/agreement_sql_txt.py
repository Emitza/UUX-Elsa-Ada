__author__ = 'elsabakiu'

import MySQLdb
import arff
import math

db = MySQLdb.connect('localhost', 'root', 'root', 'feature_extraction')
# prepare a cursor object using cursor() method
cursor = db.cursor()

# execute SQL query using execute() method.
sentences_sql = "SELECT sentence, AVG(score1) s1, AVG(score2) s2 FROM `agreements` GROUP BY sentence"
try:
   # Execute the SQL command
   cursor.execute(sentences_sql)
   sentences = cursor.fetchall()

   f = open('../data_txt/app_store_data/agreements_truth_dataset.txt', 'r+')

   for sentence in sentences:
      sent = sentence[0].replace("\n", " ")
      sent = sent.replace("\t", " ")
      if (sentence[1] == None):
          score_avg = int(sentence[2])
      elif (sentence[2] == None):
          score_avg = int(sentence[1])
      else:
          score_avg = int((int(sentence[1]) + int(sentence[2])) / 2)

      f.write(str(score_avg) + "\t" + sent + "\n")

   f.close()

except Exception, e:
   print e

# disconnect from server
db.close()