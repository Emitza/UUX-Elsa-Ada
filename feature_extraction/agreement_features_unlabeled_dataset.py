__author__ = 'elsabakiu'

import MySQLdb
import arff
import math

db = MySQLdb.connect('localhost', 'root', 'root', 'feature_extraction')
# prepare a cursor object using cursor() method
cursor = db.cursor()

# execute SQL query using execute() method.
sentences_sql = "SELECT DISTINCT a.sentence, r.project_id FROM `agreements` a INNER JOIN reviews r on a.review_id = r.review_id ORDER BY sentence"
try:
   # Execute the SQL command
   cursor.execute(sentences_sql)
   sentences = cursor.fetchall()

   f = open('data_txt/agreements_features_unlabeled_dataset.txt', 'r+')

   for sentence in sentences:
      sent = sentence[0].replace("\n", "")
      sent = sent.replace("\t", " ")
      f.write(sent + "\t" + str(sentence[1]) + "\n");

except Exception, e:
   print e

# disconnect from server
db.close()