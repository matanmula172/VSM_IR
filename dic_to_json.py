import json
from corpus_data import *

filenames = ["cf74.xml", "cf75.xml", "cf76.xml", "cf77.xml", "cf78.xml", "cf79.xml"]


def load_json_file(file_name):
    with open(file_name) as json_file:
        data = json.load(json_file)
    return data

def dump_json_file():
    data = CorpusData(filenames)
    doc_tf_idf_dictionary = data.tf_idf_data()
    with open("corpus_tf_idf.json", "w") as outfile:
        json.dump(doc_tf_idf_dictionary, outfile)

def dump_words_over_docs():
    data = CorpusData(filenames)
    doc_tf_idf_dictionary = data.tf_idf_data()
    word_over_docs = data.word_over_docs
    with open("words_over_docs.json", "w") as outfile:
        json.dump(word_over_docs, outfile)