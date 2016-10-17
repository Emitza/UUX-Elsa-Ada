__author__ = 'elsabakiu'

import MySQLdb
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

# Connect to the uux_data database
def uux_db_connect():
    db = MySQLdb.connect('localhost', 'root', 'root', 'labeled_uux_data')
    # prepare a cursor object using cursor() method
    cursor = db.cursor()
    return cursor

# Get the labeled sentences
def getUUXSentences(numDimensions):
    try:
        cursor = uux_db_connect()
        if(numDimensions == 22):
            sentences_sql = "SELECT DISTINCT sentence FROM labeled_sentences ORDER BY sentence_id"
        else:
            sentences_sql = "SELECT DISTINCT sentence FROM labeled_sentences WHERE dimension_id is NULL or dimension_id in ( 2, 4, 5, 8, 9, 13, 16, 17, 18, 19, 22) ORDER BY sentence_id"
        cursor.execute(sentences_sql)
        #Fetch all the rows in a list of lists.
        sentences = cursor.fetchall()
        sentences_array = []
        for sentence in sentences:
            sentences_array.append(sentence[0].decode('string_escape'))
        return np.array(sentences_array)
    except Exception, e:
        print e

# Get the dimensions
def getUUXDimensions(numDimensions):
    try:
        cursor = uux_db_connect()
        if(numDimensions == 22):
            dimension_sql = "SELECT dimension FROM uux_dimensions"
        else:
            dimension_sql = "SELECT dimension FROM uux_dimensions WHERE dimension in ( 'Learnability', 'Errors_Effectiveness', 'Satisfaction', 'Hedonic', 'Detailed_usability', 'Pleasure', 'Affect_Emotion', 'Enjoyment_Fun', 'Aesthetics_Appeal', 'Engagement_Flow', 'Frustration')"
        cursor.execute(dimension_sql)
        #Fetch all the rows in a list of lists.
        dimensions = cursor.fetchall()
        dimensions_array = []
        for dimension in dimensions:
            dimensions_array.append(dimension[0])
        return dimensions_array
    except Exception, e:
        print e

# Get the dimensions for each sentence
def getUUXSentenceDimension(numDimensions):
    try:
        cursor = uux_db_connect()
        if(numDimensions == 22):
            sentences_sql = "SELECT DISTINCT sentence_id FROM labeled_sentences ORDER BY sentence_id"
            sentence_dimension_sql = "SELECT dimension_id FROM labeled_sentences WHERE sentence_id = %s"
        else:
            sentences_sql = "SELECT DISTINCT sentence_id FROM labeled_sentences WHERE dimension_id is NULL or dimension_id in ( 2, 4, 5, 8, 9, 13, 16, 17, 18, 19, 22)"
            sentence_dimension_sql = "SELECT dimension_id FROM labeled_sentences WHERE sentence_id = %s AND dimension_id in ( 2, 4, 5, 8, 9, 13, 16, 17, 18, 19, 22)"

        cursor.execute(sentences_sql)
        sentence_ids = cursor.fetchall()
        sentence_dimensions = []
        for sentence_id in sentence_ids:
            cursor.execute(sentence_dimension_sql, (sentence_id))
            dimension_ids = cursor.fetchall()
            row = []
            for dimension_id in dimension_ids:
                if(not dimension_id[0] is None):
                    row.append(dimension_id[0])
                else:
                    row = []
            sentence_dimensions.append(row)
        return np.asarray(sentence_dimensions)
    except Exception, e:
        print e


# Get the labeled sentences
def getUUXSentenceIds(numDimensions):
    try:
        cursor = uux_db_connect()
        if(numDimensions == 22):
            sentences_sql = "SELECT DISTINCT sentence_id FROM labeled_sentences ORDER BY sentence_id"
        else:
            sentences_sql = "SELECT DISTINCT sentence_id FROM labeled_sentences WHERE dimension_id is NULL or dimension_id in ( 2, 4, 5, 8, 9, 13, 16, 17, 18, 19, 22)"
        cursor.execute(sentences_sql)
        #Fetch all the rows in a list of lists.
        sentences = cursor.fetchall()
        sentences_array = []
        for sentence in sentences:
            sentences_array.append(sentence[0])
        return np.array(sentences_array)
    except Exception, e:
        print e



# Get the labeled sentences
def getBinaryLabels(numDimensions):
    try:
        cursor = uux_db_connect()
        if(numDimensions == 22):
            binary_labels_sql = "SELECT sentence_id, labelSet FROM binary_labeled_sentences ORDER BY sentence_id"
        else:
            binary_labels_sql = "SELECT sentence_id, labelSet FROM binary_labeled_sentences_11dim ORDER BY sentence_id"
        cursor.execute(binary_labels_sql)
        #Fetch all the rows in a list of lists.
        labels = cursor.fetchall()
        sentences_ids = []
        labels_array = []
        for label in labels:
            sentences_ids.append(label[0])
            labels_array.append(label[1])
        return np.array(sentences_ids), np.array(labels_array)
    except Exception, e:
        print e



#Insert the dimensions' binary matrix to database
def insertBinaryLabels_DB():
    try:
        db = MySQLdb.connect('localhost', 'root', 'root', 'labeled_uux_data')
        # prepare a cursor object using cursor() method
        cursor = db.cursor()

        sentence_ids = getUUXSentenceIds(22)
        sentence_dimensions = getUUXSentenceDimension(22)
        labels_binary = MultiLabelBinarizer().fit_transform(sentence_dimensions)

        for i in range(0, len(sentence_ids)):
            sql = "INSERT INTO binary_labeled_sentences VALUES (NULL ,%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
            singleBinaryLabel = ''.join(str(label) for label in labels_binary[i])
            data = (sentence_ids[i], labels_binary[i][0], labels_binary[i][1], labels_binary[i][2], labels_binary[i][3], labels_binary[i][4], labels_binary[i][5], labels_binary[i][6], labels_binary[i][7], labels_binary[i][8], labels_binary[i][9], labels_binary[i][10], labels_binary[i][11], labels_binary[i][12], labels_binary[i][13], labels_binary[i][14], labels_binary[i][15], labels_binary[i][16], labels_binary[i][17], labels_binary[i][18], labels_binary[i][19], labels_binary[i][20], labels_binary[i][21], singleBinaryLabel)
            cursor.execute(sql, data)
        db.commit()
        cursor.close()
        db.close()

    except Exception, e:
        print e


#Insert the dimensions' binary matrix to database
def insertBinaryLabels11Dimensions_DB():
    try:
        db = MySQLdb.connect('localhost', 'root', 'root', 'labeled_uux_data')
        # prepare a cursor object using cursor() method
        cursor = db.cursor()

        sentence_ids = getUUXSentenceIds(11)
        sentence_dimensions = getUUXSentenceDimension(11)
        labels_binary = MultiLabelBinarizer().fit_transform(sentence_dimensions)

        for i in range(0, len(sentence_ids)):
            sql = "INSERT INTO binary_labeled_sentences_11dim VALUES (NULL ,%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
            singleBinaryLabel = ''.join(str(label) for label in labels_binary[i])
            data = (sentence_ids[i], labels_binary[i][0], labels_binary[i][1], labels_binary[i][2], labels_binary[i][3], labels_binary[i][4], labels_binary[i][5], labels_binary[i][6], labels_binary[i][7], labels_binary[i][8], labels_binary[i][9], labels_binary[i][10], singleBinaryLabel)
            cursor.execute(sql, data)
        db.commit()
        cursor.close()
        db.close()

    except Exception, e:
        print e