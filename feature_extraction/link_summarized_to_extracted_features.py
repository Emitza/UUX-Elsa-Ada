#! /usr/bin/env python
# Date: 06.05.2014
# Parameters: project name
import re
import string
import sys
import nltk
from decimal import *
from nltk.collocations import *
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import wordnet
from collections import Counter
from collections import OrderedDict
from StringIO import StringIO
from nltk.corpus import wordnet as wn
from itertools import product
from nltk.corpus import stopwords
from nltk.tag.simplify import simplify_wsj_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.collocations import *
import MySQLdb
import difflib
import itertools as IT

f = open('Output.txt', 'w')

db = MySQLdb.connect(unix_socket = "/Applications/MAMP/tmp/mysql/mysql.sock",
                     host="localhost",
                     user="root",
                     passwd="root",
                     db="labeled_uux_data")
    
cur = db.cursor()

# the following function does a simple string comparison 
def same(seq1, seq2):
    return difflib.SequenceMatcher(a=seq1.lower(), b=seq2.lower()).ratio() >0.99

cur.execute('select id,name from automatically_extracted_features')
data = cur.fetchall()

cur.execute('select id,name from summarized_features')
summarized_features = cur.fetchall()

for d in data:
	for s in summarized_features:		
		if (same(d[1], s[1])):
			print d[1]
			cur.execute(' update automatically_extracted_features set summarized_feature_id = %s where name like %s' ,(s[0], d[1]))
			db.commit()
			break;
			
cur.close()
db.close()
