__author__ = 'elsabakiu'

import HTMLParser
from whoosh.fields import *
from whoosh.index import create_in
from whoosh.qparser import QueryParser
from whoosh import qparser, query
from whoosh.analysis import StemmingAnalyzer

f = open('data_txt/agreements_features_truth_dataset.txt')
features = f.readlines()
f.close()

f = open('data_txt/agreements_features_sentistrength_optimized.txt')
svm_predictions = f.readlines()
f.close()

file = open('data_txt/agreements_features_aggregated_sentistrength.txt', 'r+')

parser = HTMLParser.HTMLParser()
stem_analyzer = StemmingAnalyzer()
#create the schema and the index for the information retrieval part - finding the sentence of each feature
schema = Schema(content=TEXT(stored=True, analyzer=stem_analyzer), sentiment=NUMERIC(stored=True), project_id=NUMERIC(stored=True))

ix = create_in("information_retrieval_files", schema)
writer = ix.writer()

for pred in svm_predictions:
    parts = pred.rstrip('\r\n').split("\t")
    writer.add_document(content=unicode(parts[1]), sentiment=int(parts[0]), project_id=int(parts[2]))

writer.commit()

for feat in features:
    f = feat.rstrip('\r\n').split("\t")
    feature = f[2]
    project_id = int(f[3])
    sent = 0
    sentences_feature = 0
    if(feature == "size limit"):
        print "stop"
    with ix.searcher() as searcher:
        query_content = QueryParser("content", ix.schema, group=qparser.AndGroup).parse(unicode(feature))
        query_project = QueryParser("project_id", ix.schema).parse(str(project_id))
        results = searcher.search(query.And([query_content, query_project]), limit= 3)
        # for r in results:
        #     print r["content"] + " " + str(r.score) + " " + str(r["sentiment"])
        if(len(results) != 0):
            sentences_feature = sum(row["sentiment"] for row in results)/results.scored_length()
            #sentences_feature = results[0]["sentiment"]
        else:
            print "No results for feature: " + feature
    file.write(str(sentences_feature) + "\t" + feature + "\n")

file.close()
