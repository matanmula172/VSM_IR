import os.path
import nltk
import numpy as np
import json
import lxml.html

from corpus_data import *
filenames = ["cf74.xml", "cf75.xml", "cf76.xml", "cf77.xml", "cf78.xml", "cf79.xml"]
data = CorpusData(filenames)
tf_idf_mat = data.tf_idf_data()
print(tf_idf_mat)

