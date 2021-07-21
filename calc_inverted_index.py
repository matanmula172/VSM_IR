import math
import xml.etree.ElementTree as ET
import os
from nltk_functions import *


def cosine_similarity(dic1, dic2, all_words_list):
    nominator, denominator1, denominator2 = 0, 0, 0
    for word in all_words_list:
        if word in dic1 and word in dic2:
            nominator += dic1[word] * dic2[word]
        if word in dic1:
            denominator1 += dic1[word] * dic1[word]
        if word in dic2:
            denominator2 += dic2[word] * dic2[word]
    return nominator / math.sqrt(denominator1 * denominator2)


def rank_documents(doc_tf_idf_dictionary, tf_idf_query, all_words_list):
    doc_keys = list(doc_tf_idf_dictionary.keys())
    similarity_arr = []
    for key in doc_keys:
        similarity = cosine_similarity(doc_tf_idf_dictionary[key], tf_idf_query, all_words_list)
        similarity_arr.append([similarity, key])
    similarity_arr.sort(key=lambda similarity_arr: similarity_arr[0], reverse=True)
    return similarity_arr


def get_text_from_document(doc):
    title = doc.findall("./TITLE")[0].text
    summary = " "
    if len(doc.findall("./EXTRACT")) > 0:
        summary += doc.findall("./EXTRACT")[0].text
    if len(doc.findall("./ABSTRACT")) > 0:
        summary += " " + doc.findall("./ABSTRACT")[0].text

    text = title + summary
    return text


def create_frequency_corpus(tokenized_text):
    corpus = dict()
    for token in tokenized_text:
        if token not in corpus:
            corpus[token] = 1
        else:
            corpus[token] += 1
    return corpus


def get_tf_ij_value(tokenized_text):
    num_words = len(tokenized_text)
    tf_dic = dict()
    corpus = create_frequency_corpus(tokenized_text)
    # normalized_factor = max(corpus.values())
    for i in range(num_words):
        tf_dic[tokenized_text[i]] = (float(corpus[tokenized_text[i]]) / float(num_words))
    return tf_dic


class CorpusData:
    def __init__(self, file_names, path):
        self.num_documents = 0
        self.frequency_corpus = dict()
        self.file_names = file_names
        self.path = path
        self.word_over_docs = dict()

    def get_words_in_corpus(self):
        return list(self.word_over_docs.keys())

    def get_word_idf_val(self, word):
        if word not in self.word_over_docs:
            return 0
        return math.log(float(self.num_documents) / float(self.word_over_docs[word]))

    def get_document_weight_arr(self, tf_dic):
        tf_idf_dic = dict()
        for word in tf_dic.keys():
            tf_idf_dic[word] = (tf_dic[word] * self.get_word_idf_val(word))
        return tf_idf_dic

    def init_idf_data(self):
        for filename in self.file_names:
            tree = ET.parse(os.path.join(self.path, filename))
            root = tree.getroot()
            documents = root.findall("./RECORD")
            for doc in documents:
                self.num_documents += 1
                text = get_text_from_document(doc)
                text = text.replace('\n', ' ')
                tokens = set(stem_tokenized_text(remove_stop_words(tokenize_text_no_punctuation(text))))
                for token in tokens:
                    if token not in self.word_over_docs:
                        self.word_over_docs[token] = 1
                    else:
                        self.word_over_docs[token] += 1

    def tf_idf_data(self):
        self.init_idf_data()
        doc_tf_idf_dictionary = dict()
        for filename in self.file_names:
            tree = ET.parse(os.path.join(self.path, filename))
            root = tree.getroot()
            documents = root.findall("./RECORD")
            for doc in documents:
                doc_id = doc.findall("./RECORDNUM")[0].text.strip()
                doc_id = str(int(doc_id))
                text = get_text_from_document(doc)
                text = text.replace('\n', ' ')
                tokenized_text = stem_tokenized_text(remove_stop_words(tokenize_text_no_punctuation(text)))

                tf_dic = get_tf_ij_value(tokenized_text)
                tf_dic = self.get_document_weight_arr(tf_dic)
                doc_tf_idf_dictionary[doc_id] = tf_dic
        return doc_tf_idf_dictionary
