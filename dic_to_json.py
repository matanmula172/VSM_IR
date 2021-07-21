import json
from calc_inverted_index import *

filenames = ["cf74.xml", "cf75.xml", "cf76.xml", "cf77.xml", "cf78.xml", "cf79.xml"]


def load_json_file(file_path):
    with open(file_path) as json_file:
        data = json.load(json_file)
    return data


def dump_json_file(path):
    data = CorpusData(filenames, path)
    doc_tf_idf_dictionary = data.tf_idf_data()
    with open("vsm_inverted_index.json", "w") as outfile:
        json.dump(doc_tf_idf_dictionary, outfile)


def dump_words_over_docs():
    data = CorpusData(filenames, None)
    doc_tf_idf_dictionary = data.tf_idf_data()
    word_over_docs = data.word_over_docs
    with open("words_over_docs.json", "w") as outfile:
        json.dump(word_over_docs, outfile)
