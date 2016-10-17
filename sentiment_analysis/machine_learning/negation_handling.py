__author__ = 'elsabakiu'

import nltk
from nltk.stem.snowball import SnowballStemmer

negation_words = ['no', 'not', 'n\'t', 'never', 'less', 'without', 'barley', 'hardly', 'rarely', 'cannot']
negation_terms = ['no longer', 'no more', 'no way', 'no where', 'by no means', 'at no time', 'not...anymore']

#stopwords = [ 'a', 'an', 'the',  'of', 'at', 'by', 'for', 'with', 'about', 'into', 'through', 'to', 'from', 'up', 'down','in', 'out', 'over', 'under', 'again', 'further']

stopwords = [ 'a', 'an',  'the', 'of', 'at', 'by', 'to', 'me', 'i', 'upon', ',', '.', '!','and', 'since', 'even', 'into', 'in', 'it', 'that', 'with', 'my', 'only', 'your',
              'mine', 'our', 'their', 'for', 'on', 'or', '\'s', 'as', 'just', '\"', 'has', 'have', 'you', 'yo', 'that', 'one' ]

nltk_stopwords = nltk.corpus.stopwords.words('english')
stemmer = SnowballStemmer("english")

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item).lower())
    return stemmed


def tokenize(text):

    tokens = nltk.word_tokenize(text)
    tokens = [token for token in tokens if token.lower() not in stopwords]

    for i in range(0, len(tokens) - 1):
        if tokens[i] in negation_words:
            if(tokens[i] + " " + tokens[i+1] in negation_terms):
                i = i + 1
            i = i + 1
            j = 0
            while(i < len(tokens) and j < 2):
                if(tokens[i] not in stopwords):
                    tokens[i] = "not_" + tokens[i]
                    j = j + 1
                i = i + 1
    stems = stem_tokens(tokens, stemmer)

    return stems

# def tokenize(text):
#
#     tokens = nltk.word_tokenize(text)
#     tokens = [token for token in tokens if token.lower() not in stopwords]
#
#     for i in range(0, len(tokens) - 1):
#         if tokens[i] + " " + tokens[i + 1] in negation_terms:
#             if(i + 3 < len(tokens)):
#                 tokens[i+2], tokens[i+3] = "not_" + tokens[i+2], "not_" + tokens[i+3]
#                 i = i + 4
#                 print tokens
#             elif(i + 2 < len(tokens)):
#                 tokens[i+2] = "not_" + tokens[i+2]
#                 i = i + 3
#                 print tokens
#         elif tokens[i] in negation_words or 'n\'t' in tokens[i]:
#             if(i + 2 < len(tokens)):
#                 tokens[i+1], tokens[i+2] = "not_" + tokens[i+1], "not_" + tokens[i+2]
#                 i = i + 3
#                 print tokens
#             else:
#                 tokens[i+1] =  "not_" + tokens[i+1]
#                 i = i + 2
#                 print tokens
#
#
#     stems = stem_tokens(tokens, stemmer)
#
#     # bigrams = BigramCollocationFinder.from_words(text)
#     # bigram_measures = nltk.collocations.BigramAssocMeasures()
#     # bigrams = bigrams.nbest(bigram_measures.pmi, 10000)
#
#     return stems

