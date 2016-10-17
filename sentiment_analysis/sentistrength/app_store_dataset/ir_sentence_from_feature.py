__author__ = 'elsabakiu'

import MySQLdb
import nltk
import HTMLParser
from whoosh.fields import *
from whoosh.index import create_in
from whoosh.qparser import QueryParser
from whoosh import qparser
from whoosh.analysis import StemmingAnalyzer


db = MySQLdb.connect('localhost', 'root', 'root', 'feature_extraction')
# prepare a cursor object using cursor() method
cursor = db.cursor()

# execute SQL query using execute() method.
agreement_sql = "SELECT r.comment text, a.label1, a.label2, a.score1,  a.score2, a.id, r.title  FROM reviews r INNER JOIN agreements a ON r.review_id = a.review_id"
try:
   # Execute the SQL command
   cursor.execute(agreement_sql)
   agreements = cursor.fetchall()

   parser = HTMLParser.HTMLParser()
   stem_analyzer = StemmingAnalyzer()
   #create the schema and the index for the information retrieval part - finding the sentence of each feature
   schema = Schema(content=TEXT(stored=True, analyzer=stem_analyzer))

   for agreement in agreements:

       ix = create_in("information_retrieval_files", schema)

       #Ask Emitza about the encoding used and replace utf-8 with the right one
       review = agreement[0]
       try:
        review = agreement[0].encode('ascii', 'ignore')
       except Exception, e:
        continue

       #HTML parser unescapes the review
       review = parser.unescape(review)
       sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
       sentences = sent_detector.tokenize(review)

       writer = ix.writer()

       for sentence in sentences:
        writer.add_document(content=unicode(sentence))

       feature = agreement[1] if agreement[1] is not None else agreement[2]

       if(feature == "size limit"):
           print "stop"

       writer.commit()
       sentences_feature = ""
       with ix.searcher() as searcher:
        query = QueryParser("content", ix.schema, group=qparser.OrGroup).parse(unicode(feature))
        results = searcher.search(query)
        if(len(results) == 0):
            sentences_feature = agreement[6]
        else:
            sentences_feature = results[0]["content"]

        # Prepare SQL query to INSERT a record into the database.
        sql = "UPDATE agreements SET sentence = (%s) WHERE id = (%s) "
        try:
            # Execute the SQL command
            cursor.execute(sql, (str(sentences_feature).encode('ascii','ignore'), agreement[5]))
            # Commit your changes in the database
            db.commit()
        except Exception, e:
            # Rollback in case there is any error
            print e
            db.rollback()

except Exception, e:
   print e

# disconnect from server
db.close()