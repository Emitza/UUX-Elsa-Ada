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

f = open('Bigrams.txt', 'w')

#########################################################################################################################
# This is the ngram extraction part of the program
db = MySQLdb.connect(unix_socket = "/Applications/MAMP/tmp/mysql/mysql.sock",
                     host="localhost",
                     user="root",
                     passwd="root",
                     db="labeled_uux_data")
    
cur = db.cursor()

EmotionLookupTable = open("EmotionLookupTable.txt","r")
senti_words= EmotionLookupTable.read()
tokenizer = RegexpTokenizer(r'\w+')
senti_tokens = tokenizer.tokenize(senti_words)

cur.execute('select sentence from sentences')# +' limit 10')
data = cur.fetchall()
data_str = "".join("%s" % x for x in data)
data_str = data_str.lower()
tokenizer = RegexpTokenizer(r'\w+')
tokens = tokenizer.tokenize(data_str)
tagged = nltk.pos_tag(tokens)

# ---------------  nouns contain only nouns, nouns_verbs contains nouns and verbs, njv contain nouns, adjectives and verbs,  ---------------  
nouns = [word for word,pos in tagged if pos == 'NN' or pos == 'NNS' or pos == 'NNP' or pos == 'NNPS']
njv= [word for word,pos in tagged if pos == 'NN' or pos == 'NNS' or pos == 'NNP' or pos == 'NNPS' or pos == 'JJ' or pos == 'JJR' or pos == 'RB' or pos == 'VB' or pos == 'VBD' or pos == 'VBG' or pos == 'VBN' or pos == 'VBZ']
nouns_verbs= [word for word,pos in tagged if pos == 'NN' or pos == 'NNS' or pos == 'NNP' or pos == 'NNPS' or pos == 'VB' or pos == 'VBD' or pos == 'VBG' or pos == 'VBN' or pos == 'VBZ']
bigram_measures = nltk.collocations.BigramAssocMeasures()
stop = stopwords.words('english')
filtered_collacation = [w for w in tokens if not w in stopwords.words('english')]

bgm = nltk.collocations.BigramAssocMeasures()
finder = nltk.collocations.BigramCollocationFinder.from_words(filtered_collacation)
finder.apply_freq_filter(3)
sign_filter = ['\'', '-', '_', '=', ':-', '1','2','3','4','5','6','7','8','9','10']
finder.apply_word_filter(lambda w: w in ('\'', '-', '_', '=', ':-', '1','2','3','4','5','6','7','8','9','10'))
false_words= ['com','app','apps','better','dropbox','tripadvisor','angrybirds','pinterest','whatsapp','picsart','evernote','cant','doesn','wouldn','shouldn','plz','gud','pls','xls','won','dont']
best_pmi = list()
for i in finder.score_ngrams(bigram_measures.pmi):
    if i[1]>=0:
		best_pmi  = best_pmi + [i[0]]

scored = finder.score_ngrams(bgm.likelihood_ratio)

bigrams = list()
counts = Counter(tokens)
false_senti_words = []

# filtering the sentiment words

for s in scored:
	for senti_token in senti_tokens:
		if s[0][0].startswith(senti_token):			
			false_senti_words.append(s[0][0])
		elif s[0][1].startswith(senti_token):			
			false_senti_words.append(s[0][1])

# applying POS tag patterns

for s in scored:
	if s[0] in best_pmi and (s[0][0] in nouns or s[0][1] in nouns) and s[0][0] in njv and s[0][1] in njv and (s[0][0] not in (false_words) and s[0][1] not in (false_words)) and (len(s[0][0])>2 and len(s[0][1])>2) and ((s[0][0] not in false_senti_words) and (s[0][1] not in false_senti_words)):	
		bigrams  = bigrams + [s[0]]

for s in bigrams:
	f.write(s[0] + " " + s[1] + "\n")

cur.close()
db.close()
