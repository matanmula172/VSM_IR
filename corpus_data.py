import math
import xml.etree.ElementTree as ET
import os
from nltk_functions import *

FILE = "cfc-xml_corrected"


class CorpusData:
    def __init__(self, file_names):
        self.num_documents = 0
        self.frequency_corpus = dict()
        self.file_names = file_names
        self.word_over_docs = dict()

    def get_word_idf_val(self, word):
        return math.log2(float(self.num_documents) / self.word_over_docs[word])

    def get_document_weight_arr(self, tokenized_document, tf_dic):
        doc_arr = []
        words = list(self.word_over_docs.keys())
        for i in range(len(words)):
            if words[i] not in tokenized_document:
                doc_arr.append(0)
            else:
                tf_value = tf_dic[words[i]]
                doc_arr.append(tf_value*self.get_word_idf_val(words[i]))
        return doc_arr

    def add_words_to_frequency_corpus(self, tokenized_text, corpus):
        for token in tokenized_text:
            if token not in corpus:
                corpus[token] = 1
            else:
                corpus[token] += 1
        return corpus

    def get_tf_ij(self, tokenized_text):
        num_words = len(tokenized_text)
        tf_dic = dict()
        corpus = dict()
        corpus = self.add_words_to_frequency_corpus(tokenized_text, corpus)
        for i in range(num_words):
            tf_dic[tokenized_text[i]] = (float(corpus[tokenized_text[i]]) / float(num_words))
        return tf_dic

    def init_idf_data(self):
        for filename in self.file_names:
            tree = ET.parse(os.path.join(FILE, filename))
            root = tree.getroot()
            documents = root.findall("./RECORD")
            for doc in documents:
                self.num_documents += 1
                title = doc.findall("./TITLE")[0].text
                summary = ""
                if len(doc.findall("./EXTRACT")) > 0:
                    summary = doc.findall("./EXTRACT")[0].text
                elif len(doc.findall("./ABSTRACT")) > 0:
                    summary = doc.findall("./ABSTRACT")[0].text

                text = title + summary
                tokens = set(stem_tokenized_text(remove_stop_words(tokenize_text_no_punctuation(text))))
                for token in tokens:
                    if token not in self.word_over_docs:
                        self.word_over_docs[token] = 1
                    else:
                        self.word_over_docs[token] += 1

    def tf_idf_data(self):
        self.init_idf_data()
        tf_idf_mat = []
        for filename in self.file_names:
            tree = ET.parse(os.path.join(FILE, filename))
            root = tree.getroot()
            documents = root.findall("./RECORD")
            for doc in documents:
                self.num_documents += 1
                doc_id = doc.findall("./RECORDNUM")[0].text.strip()
                doc_id = str(int(doc_id))
                title = doc.findall("./TITLE")[0].text
                summary = ""
                if len(doc.findall("./EXTRACT")) > 0:
                    summary = doc.findall("./EXTRACT")[0].text
                elif len(doc.findall("./ABSTRACT")) > 0:
                    summary = doc.findall("./ABSTRACT")[0].text
                text = title + summary
                tokenized_text = stem_tokenized_text(remove_stop_words(tokenize_text_no_punctuation(text)))
                tf_dic = dict()
                tf_dic = self.get_tf_ij(tokenized_text)
                doc_array = self.get_document_weight_arr(tokenized_text, tf_dic)
                tf_idf_mat.append(doc_array)
        return tf_idf_mat
