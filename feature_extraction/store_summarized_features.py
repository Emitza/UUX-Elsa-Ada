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
sys.stdout = f

db = MySQLdb.connect(unix_socket = "/Applications/MAMP/tmp/mysql/mysql.sock",
                     host="localhost",
                     user="root",
                     passwd="root",
                     db="labeled_uux_data")
    
cur = db.cursor()

# the following function does a simple string comparison 
def same(seq1, seq2):
    return difflib.SequenceMatcher(a=seq1.lower(), b=seq2.lower()).ratio() >0.99
	
cur.execute('select distinct(name) from automatically_extracted_features order by name')
data = cur.fetchall()
review_IDs = []
check = 0	
tokenizer = RegexpTokenizer(r'\w+')
summarized_feature_names = []
summarized_feature_IDs = []
index= 0
for d1 in data:	
	bigram_tokens_1 = tokenizer.tokenize(d1[0])		
	for d2 in data:
		bigram_tokens_2 = tokenizer.tokenize(d2[0])			
		if d2[0] not in summarized_feature_names and not ((same(bigram_tokens_1[0], bigram_tokens_2[0]) and same(bigram_tokens_1[1], bigram_tokens_2[1])) or ((bigram_tokens_1[1], bigram_tokens_2[0]) and same(bigram_tokens_1[0], bigram_tokens_2[1]))):			
			summarized_feature_names.append(d2[0])
			
summarized_features = []
duplicate_pairs = []
duplicates_first_element = []

duplicates = []
for s1 in duplicate_pairs:
	for s2 in duplicate_pairs:
		temp1 = s1.split(',')
		temp2 = s2.split(',')
		if temp1[0]== temp2[1] and temp1[1]== temp2[0] and temp1[0] not in duplicates and temp1[1] not in duplicates and temp2[0] not in duplicates  and temp2[1] not in duplicates :
			duplicates.append(temp1[0])
# add all the features that are not duplicates
for s in summarized_feature_names:
	if s not in duplicates_first_element:		
		summarized_features.append(s)
		index = index+1
		summarized_feature_IDs.append(index)
# in case many features have same names, then add only one of the many matches
for s in duplicates:	
	summarized_features.append(s)
	index = index+1
	summarized_feature_IDs.append(index)

summarized_features_with_IDs = zip(summarized_feature_IDs,summarized_features)

for s in summarized_features_with_IDs:
	print s
	cur.execute('INSERT INTO  summarized_features (`name`) values (%s) ',(s[1]))
	db.commit()
			
cur.close()
db.close()
