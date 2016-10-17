__author__ = 'elsabakiu'

import nltk
import nltk.stem.snowball

def tokenize(text):

    tokens = nltk.word_tokenize(text)

    for i in range(0, len(tokens) - 1):
        if tokens[i] in negation_words:
            if(tokens[i] + " " + tokens[i+1] in negation_terms):
                i = i + 1
                print "Negation Phrase: " + tokens[i-1] + " " + tokens[i]
            i = i + 1
            j = 0
            while(i < len(tokens) and j < 2):
                if(tokens[i] not in stopwords):
                    tokens[i] = "not_" + tokens[i]
                    j = j + 1
                i = i + 1
    return ' '.join(tokens).replace(" n\'t", "n\'t")


negation_words = ['no', 'not', 'n\'t', 'never', 'less', 'without', 'barley', 'hardly', 'rarely', 'cannot']
negation_terms = ['no longer', 'no more', 'no way', 'no where', 'by no means', 'at no time', 'not...anymore']

stopwords = [ 'a', 'an',  'the', 'of', 'at', 'by', 'to', 'me', 'i', 'upon', ',', '.', '!','and', 'since', 'even', 'into', 'in', 'it', 'that', 'with', 'my', 'only', 'your',
              'mine', 'our', 'their', 'for', 'on', 'or', '\'s', 'as', 'just', '\"', 'has', 'have', 'you', 'yo', 'that', 'one' ]


f = open('../sentistrength/data_txt/app_store_data/sentences.txt')
lines = f.readlines()
f.close()

f = open('data_txt/sentences_negated.txt', 'w')


for line in lines:
    row = []
    elements = line.rstrip('\r\n').split('\t')
    sentence = elements[1]
    sentiment = elements[0]

    negated_sentence = tokenize(sentence)
    print negated_sentence
    f.write(sentiment + "\t" + negated_sentence + "\n")

