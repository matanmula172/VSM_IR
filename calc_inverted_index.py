import math
import xml.etree.ElementTree as ET
import os
from nltk_functions import *


# given two tf_idf dictionaries representing two texts, and a list of all the words in the corpus,
# this functions returns the cosine similarity of the texts
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


# given tf_idf dictionary of all the texts in the corpus, a tf_idf dictionary of the query,
# representing two texts and a list of all the words in the corpus,
# this functions returns a tuple array - each entry is [cosine similarity value, document_id]
# the array is sorted by the similarity
def rank_documents(doc_tf_idf_dictionary, tf_idf_query, all_words_list):
    doc_keys = list(doc_tf_idf_dictionary.keys())
    similarity_arr = []
    for key in doc_keys:
        similarity = cosine_similarity(doc_tf_idf_dictionary[key], tf_idf_query, all_words_list)
        similarity_arr.append([similarity, key])
    similarity_arr.sort(key=lambda similarity_arr: similarity_arr[0], reverse=True)
    return similarity_arr


# Given an Etree document, this function returns all the text from the TITLE, EXTRACT, ABSTRACT tags,
# if there is any
def get_text_from_document(doc):
    title = doc.findall("./TITLE")[0].text
    summary = " "
    if len(doc.findall("./EXTRACT")) > 0:
        summary += doc.findall("./EXTRACT")[0].text
    if len(doc.findall("./ABSTRACT")) > 0:
        summary += " " + doc.findall("./ABSTRACT")[0].text

    text = title + summary
    return text


# Given a tokenized text, this function returns a frequency dictionary
def create_frequency_corpus(tokenized_text):
    corpus = dict()
    for token in tokenized_text:
        if token not in corpus:
            corpus[token] = 1
        else:
            corpus[token] += 1
    return corpus


# Given a tokenized text, this function returns a dictionary,
# keys are the words of the text, and values are the tf_ij value
def get_tf_ij_value(tokenized_text):
    num_words = len(tokenized_text)
    tf_dic = dict()
    corpus = create_frequency_corpus(tokenized_text)
    for i in range(num_words):
        tf_dic[tokenized_text[i]] = (float(corpus[tokenized_text[i]]) / float(num_words))
    return tf_dic


class CorpusData:
    def __init__(self, file_names, path):
        # after init, equals number of all documents in the corpus
        self.num_documents = 0
        # equals filenames
        self.file_names = file_names
        self.path = path
        # after init, keys are all words in the corpus,
        # and values are how many documents each word is in
        self.word_over_docs = dict()

    def get_words_in_corpus(self):
        return list(self.word_over_docs.keys())

    def get_word_idf_val(self, word):
        if word not in self.word_over_docs:
            return 0
        return math.log(float(self.num_documents) / float(self.word_over_docs[word]))

    # Given a text that has been processed into a tf_ij dictionary,
    # this function returns a dictionary where each word value is the tf_idf value
    def get_document_weight_arr(self, tf_dic):
        tf_idf_dic = dict()
        for word in tf_dic.keys():
            tf_idf_dic[word] = (tf_dic[word] * self.get_word_idf_val(word))
        return tf_idf_dic

    # This function initializes self.word_over_docs, self.num_documents
    def init_idf_data(self):
        for filename in self.file_names:
            tree = ET.parse(os.path.join(self.path, filename))
            root = tree.getroot()
            documents = root.findall("./RECORD")
            for doc in documents:
                # count documents
                self.num_documents += 1
                text = get_text_from_document(doc)
                text = text.replace('\n', ' ')
                # count each token once per document
                tokens = set(stem_tokenized_text(remove_stop_words(tokenize_text_no_punctuation(text))))
                for token in tokens:
                    if token not in self.word_over_docs:
                        self.word_over_docs[token] = 1
                    else:
                        self.word_over_docs[token] += 1

    # This function returns tf_idf_dictionary of all corpus, the initialized self.word_over_docs, self.num_documents
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

                # get tf_ij value for each word in this document
                tf_dic = get_tf_ij_value(tokenized_text)
                # get tf_idf value for each word in this document
                tf_dic = self.get_document_weight_arr(tf_dic)
                doc_tf_idf_dictionary[doc_id] = tf_dic
        return doc_tf_idf_dictionary, self.word_over_docs, self.num_documents
