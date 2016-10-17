__author__ = 'elsabakiu'

import MySQLdb
import skll
from nltk.metrics.agreement import AnnotationTask

db = MySQLdb.connect('localhost', 'root', 'root', 'labeled_uux_data')
# prepare a cursor object using cursor() method
cursor = db.cursor()

# execute SQL query using execute() method.
sentences_sql_reviewer1 = "SELECT sentence, positive_sentiment, negative_sentiment, id, 10_scale, 5_scale FROM sentiment_test_sentences_reviewers WHERE reviewer_id in (4) ORDER BY sentence_id"
sentences_sql_reviewer2 = "SELECT sentence, positive_sentiment, negative_sentiment, id, 10_scale, 5_scale FROM sentiment_test_sentences_reviewers WHERE reviewer_id in (5) ORDER BY sentence_id"

i = 0

try:
   # Execute the SQL command
   cursor.execute(sentences_sql_reviewer1)
   sentences_r1 = cursor.fetchall()

   cursor.execute(sentences_sql_reviewer2)
   sentences_r2 = cursor.fetchall()

   sentiment_r1 = []
   sentiment_r2 = []

   data = []

   sentiment_r1_5_scale = []
   sentiment_r2_5_scale = []

   for r1, r2 in zip(sentences_r1, sentences_r2):

      sentiment_r1_5_scale.append(int(r1[5]))
      data.append((6, r1[0], r1[5]))

      sentiment_r2_5_scale.append(int(r2[5]))
      data.append((7, r2[0], r2[5]))

      if (r1[0] != r2[0]):
          print r1[0]

except Exception, e:
   print e

# disconnect from server
db.close()
print i

print skll.kappa(sentiment_r1_5_scale, sentiment_r2_5_scale)

annotation = AnnotationTask(data=data)

print annotation.kappa()
print annotation.alpha()
