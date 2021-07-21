import json
from calc_inverted_index import *

filenames = ["cf74.xml", "cf75.xml", "cf76.xml", "cf77.xml", "cf78.xml", "cf79.xml"]


def load_json_file(file_path):
    with open(file_path) as json_file:
        data = json.load(json_file)
    return data


def dump_json_file(path):
    data = CorpusData(filenames, path)
    doc_tf_idf_dictionary, words_over_docs, num_documents = data.tf_idf_data()
    vsm_inverted_dic = dict()
    vsm_inverted_dic["tf_idf"] = doc_tf_idf_dictionary
    vsm_inverted_dic["words_over_docs"] = words_over_docs
    vsm_inverted_dic["num_documents"] = num_documents
    with open("vsm_inverted_index.json", "w") as outfile:
        json.dump(vsm_inverted_dic, outfile)


# def dump_words_over_docs():
#     data = CorpusData(filenames, None)
#     doc_tf_idf_dictionary = data.tf_idf_data()
#     word_over_docs = data.word_over_docs
#     with open("words_over_docs.json", "w") as outfile:
#         json.dump(word_over_docs, outfile)
