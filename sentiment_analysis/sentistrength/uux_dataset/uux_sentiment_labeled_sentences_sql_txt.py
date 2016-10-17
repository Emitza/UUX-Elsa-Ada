__author__ = 'elsabakiu'

import MySQLdb
import arff

db = MySQLdb.connect('localhost', 'root', 'root', 'labeled_uux_data')
# prepare a cursor object using cursor() method
cursor = db.cursor()

# execute SQL query using execute() method.
sentences_sql = "SELECT sentence, positive_sentiment, negative_sentiment, 10_class_sentiment, 5_class_sentiment FROM sentiment_test_sentences ORDER BY sentence_id"
try:
   # Execute the SQL command
   cursor.execute(sentences_sql)
   sentences = cursor.fetchall()

   f_values = open('DataTxt/sentiment_truth_dataset.txt', 'r+')

   for sentence in sentences:
      f_values.write(str(sentence[4]) + "\t" + sentence[0].decode('string_escape') + "\n")

   f_values.close()

   f_values.close()

except Exception, e:
   print e

# disconnect from server
db.close()