import math
import os.path
import nltk
import numpy as np
import json
import lxml.html

import dic_to_json
from corpus_data import *
from dic_to_json import *

# Psuedo main

data = CorpusData(filenames)
doc_tf_idf_dictionary = dict(load_json_file("corpus_tf_idf.json"))


query = "pancreatic enzyme deficiency and abnormally high sweat"
tokens = stem_tokenized_text(remove_stop_words(tokenize_text_no_punctuation(query)))
word_over_docs = dict(load_json_file("words_over_docs.json"))
num_documents = len(list(doc_tf_idf_dictionary.keys()))

tf_dic = dict()
# calculate TFij
for i in range(len(tokens)):
    tf_dic[tokens[i]] = (float(tokens.count(tokens[i])) / float(len(tokens)))

tf_idf_query = get_query_weight_arr(tokens, tf_dic, word_over_docs, num_documents)

ranked_documents = rank_documents(doc_tf_idf_dictionary, tf_idf_query)
print(ranked_documents)