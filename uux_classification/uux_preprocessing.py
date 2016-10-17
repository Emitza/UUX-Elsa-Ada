__author__ = 'elsabakiu'

import nltk
from nltk.stem.snowball import SnowballStemmer


nltk_stopwords = nltk.corpus.stopwords.words('english')
stemmer = SnowballStemmer("english")

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item).lower())
    return stemmed

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    tokens = [token for token in tokens if token.lower() not in nltk_stopwords]
    stems = stem_tokens(tokens, stemmer)
    return stems

