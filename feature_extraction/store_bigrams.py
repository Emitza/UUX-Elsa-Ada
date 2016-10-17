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

bigrams = open("Bigrams.txt","r")
bigrams_raw= bigrams.read()
bigrams = []
for s in bigrams_raw.splitlines():
	bigrams.append(s.split())	

cur.execute('select id, sentence  from sentences')
data = cur.fetchall()
review_IDs = []
bigrams_without_IDs = []
bigrams_with_IDs = []
lmtzr = WordNetLemmatizer()
lemmatized_data_IDs = []
lemmatized_data = []
lemmatized_record = ''
tokenizer = RegexpTokenizer(r'\w+')

for d in data:			
		data_tokens = tokenizer.tokenize((d[1]).lower())
		tagged_data = nltk.pos_tag(data_tokens)		
		filtered_tagged_data = [word for word in tagged_data if not word in stopwords.words('english')]
		for word,pos in filtered_tagged_data:			
			if pos.startswith('J'):
				lemmatized_record = lemmatized_record+(lmtzr.lemmatize(word,wordnet.ADJ)+' ')	
			elif pos.startswith('V'):
				lemmatized_record = lemmatized_record+(lmtzr.lemmatize(word,wordnet.VERB)+' ')	
			elif pos.startswith('N'):
				lemmatized_record = lemmatized_record+(lmtzr.lemmatize(word,wordnet.NOUN)+' ')	
			elif pos.startswith('R'):
				lemmatized_record = lemmatized_record+(lmtzr.lemmatize(word,wordnet.ADV)+' ')	
			else:
				lemmatized_record = lemmatized_record+(lmtzr.lemmatize(word)+' ')	
		lemmatized_data.append(lemmatized_record)
		lemmatized_data_IDs.append(d[0])		
		lemmatized_record = ''	
lemmatized_data_With_IDs = zip(lemmatized_data_IDs, lemmatized_data)

	
for b in bigrams:	
	b1 = lmtzr.lemmatize(b[0])
	b2 = lmtzr.lemmatize(b[1])	
	for d in lemmatized_data_With_IDs:		
		matches =  re.search(b1+"\W+(?:\w+\W+){0,3}"+b2,d[1], re.IGNORECASE)			
		if matches==None:			
			matches =  re.search(b2+"\W+(?:\w+\W+){0,3}"+b1,d[1], re.IGNORECASE)			
		if matches:				
			review_IDs.append(d[0])
			bigrams_without_IDs.append(b)

bigrams_with_IDs =  zip(review_IDs, bigrams_without_IDs)

for idx, feature in enumerate(bigrams_with_IDs):	
	review_id = feature[0]
	name = feature[1][0]+' '+feature[1][1]	
	print idx, review_id, name, 0 , 0
	
	# insert into automatically_extracted_features_njv in case of nouns verbs and adjectives
	# insert into automatically_extracted_features_nouns_verbs in case of nouns and verbs
	# insert into automatically_extracted_features_nouns in case of nouns
	# insert into automatically_extracted_features_np in case of noun phrases
	
	cur.execute('INSERT INTO  automatically_extracted_features (`name`, `sentence_id`) values (%s, %s) ',(feature[1][0]+' '+feature[1][1], review_id))
	db.commit()
	
print 'automatically extracted features: ',len(bigrams_with_IDs)

cur.close()
db.close()
