import sys
from dic_to_json import *


# calculate query TF_IDF
def query_tf_idf(data, tokens):
    tf_dict = dict()

    for i in range(len(tokens)):
        tf_dict[tokens[i]] = (float(tokens.count(tokens[i])) / float(len(tokens)))
    return data.get_document_weight_arr(tf_dict)


def create_ranked_query_docs(ranked_documents):
    with open("ranked_query_docs.txt", "w") as ranked_query_docs:
        for doc in ranked_documents[:50]:
            ranked_query_docs.write(doc[1] + '\n')


if __name__ == '__main__':
    if sys.argv[1] == 'create_index':
        path = sys.argv[2]
        dump_json_file(path)
    else:
        query = str(sys.argv[3:])
        index_path = str(sys.argv[2])
        data = CorpusData(filenames, index_path)
        doc_tf_idf_dictionary = dict(load_json_file(index_path))
        tokens = stem_tokenized_text(remove_stop_words(tokenize_text_no_punctuation(query)))
        word_over_docs = dict(load_json_file("words_over_docs.json"))
        data.word_over_docs = word_over_docs
        data.num_documents = 1239

        tf_idf_query = query_tf_idf(data, tokens)

        ranked_documents = rank_documents(doc_tf_idf_dictionary, tf_idf_query, list(word_over_docs.keys()))

        create_ranked_query_docs(ranked_documents)
