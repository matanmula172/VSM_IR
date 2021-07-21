import json
from calc_inverted_index import *

filenames = ["cf74.xml", "cf75.xml", "cf76.xml", "cf77.xml", "cf78.xml", "cf79.xml"]

# Given a path this function opens a json file and returns the data saved in it
def load_json_file(file_path):
    with open(file_path) as json_file:
        data = json.load(json_file)
    return data

# Given a path, this function creates a CorpusData objects,
# calculates tf_idf values for each word in each text in the files (in file_names) and
# dumps the information into a json file
def dump_json_file(path):
    data = CorpusData(filenames, path)
    doc_tf_idf_dictionary, words_over_docs, num_documents = data.tf_idf_data()
    vsm_inverted_dic = dict()
    vsm_inverted_dic["tf_idf"] = doc_tf_idf_dictionary
    vsm_inverted_dic["words_over_docs"] = words_over_docs
    vsm_inverted_dic["num_documents"] = num_documents
    with open("vsm_inverted_index.json", "w") as outfile:
        json.dump(vsm_inverted_dic, outfile)
